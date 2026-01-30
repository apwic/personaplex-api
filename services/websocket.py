"""WebSocket connection management for streaming audio."""

import asyncio
import json
import os
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from .streaming import StreamingSession, create_session

API_KEY = os.getenv("API_KEY", "default-api-key-change-me")
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "4"))


class ConnectionManager:
    """Manages active WebSocket connections and their streaming sessions."""

    def __init__(self, mimi_model=None, moshi_model=None):
        self.active_connections: dict[str, WebSocket] = {}
        self.sessions: dict[str, StreamingSession] = {}
        self.gpu_lock = asyncio.Lock()
        self.mimi_model = mimi_model
        self.moshi_model = moshi_model

    def set_models(self, mimi_model, moshi_model):
        """Set the models after loading."""
        self.mimi_model = mimi_model
        self.moshi_model = moshi_model

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """Accept connection and create session."""
        if len(self.sessions) >= MAX_CONCURRENT_SESSIONS:
            await websocket.close(code=4008, reason="Session limit reached")
            return False

        if self.mimi_model is None or self.moshi_model is None:
            await websocket.close(code=4001, reason="Model not loaded")
            return False

        await websocket.accept()
        self.active_connections[session_id] = websocket

        async with self.gpu_lock:
            session = create_session(self.mimi_model, self.moshi_model, session_id)
            session.start()
            self.sessions[session_id] = session

        await self._send_event(
            websocket,
            "session_started",
            {"session_id": session_id, "frame_size": session.frame_size},
        )
        logger.info(f"WebSocket connected: {session_id}")
        return True

    async def disconnect(self, session_id: str):
        """Clean up connection and session."""
        if session_id in self.sessions:
            self.sessions[session_id].stop()
            del self.sessions[session_id]

        if session_id in self.active_connections:
            del self.active_connections[session_id]

        logger.info(f"WebSocket disconnected: {session_id}")

    async def process_audio(self, session_id: str, audio_data: bytes) -> list[bytes]:
        """Process audio frame with GPU lock."""
        if session_id not in self.sessions:
            raise RuntimeError("Session not found")

        async with self.gpu_lock:
            return self.sessions[session_id].process_frame(audio_data)

    async def reset_session(self, session_id: str):
        """Reset a session's streaming state."""
        if session_id not in self.sessions:
            raise RuntimeError("Session not found")

        async with self.gpu_lock:
            self.sessions[session_id].reset()

    async def _send_event(self, websocket: WebSocket, event: str, payload: dict = None):
        """Send JSON event to client."""
        message = {
            "type": "event",
            "event": event,
            "payload": payload or {},
            "timestamp": time.time(),
        }
        await websocket.send_text(json.dumps(message))


async def verify_websocket_auth(websocket: WebSocket) -> Optional[str]:
    """Verify WebSocket authentication via query param or first message.

    Returns session_id if authenticated, None otherwise.
    """
    token = websocket.query_params.get("token")

    if not token:
        try:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            msg = json.loads(data)
            if msg.get("type") != "auth":
                await websocket.close(code=4001, reason="Authentication required")
                return None
            token = msg.get("token")
        except asyncio.TimeoutError:
            await websocket.close(code=4001, reason="Authentication timeout")
            return None
        except json.JSONDecodeError:
            await websocket.close(code=4001, reason="Invalid auth message")
            return None

    if token != API_KEY:
        await websocket.close(code=4003, reason="Invalid token")
        return None

    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
    return session_id


async def handle_websocket(websocket: WebSocket, manager: ConnectionManager):
    """Main WebSocket handler for streaming audio."""
    await websocket.accept()

    session_id = await verify_websocket_auth(websocket)
    if session_id is None:
        return

    if not await manager.connect(websocket, session_id):
        return

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                try:
                    output_chunks = await manager.process_audio(session_id, message["bytes"])
                    for chunk in output_chunks:
                        await websocket.send_bytes(chunk)
                except Exception as e:
                    logger.error(f"Session {session_id}: processing error: {e}")
                    await manager._send_event(
                        websocket,
                        "error",
                        {"code": "E003", "message": str(e), "recoverable": True},
                    )

            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                    action = msg.get("action")

                    if action == "reset":
                        await manager.reset_session(session_id)
                        await manager._send_event(websocket, "reset_ack")

                    elif action == "stop":
                        await manager._send_event(websocket, "session_ended")
                        break

                    elif action == "ping":
                        await manager._send_event(
                            websocket, "pong", {"server_time": time.time()}
                        )

                except json.JSONDecodeError:
                    logger.warning(f"Session {session_id}: invalid JSON message")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Session {session_id}: WebSocket error: {e}")
    finally:
        await manager.disconnect(session_id)
