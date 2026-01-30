"""Real-time speech-to-speech client for PersonaPlex API."""

import asyncio
import json
import struct
import sys
import threading
import queue

import websockets

try:
    import pyaudio
except ImportError:
    print("Install pyaudio: pip install pyaudio")
    sys.exit(1)

API_URL = "wss://apwic--personaplex-api-modelapi-serve.modal.run/ws/stream"
API_TOKEN = "pa_dasklhuozchxvwqjeljAfdja"

SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80ms at 24kHz (Mimi requirement)
CHANNELS = 1
FORMAT = pyaudio.paInt16


class RealtimeClient:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.send_queue = queue.Queue()
        self.recv_queue = queue.Queue()
        self.running = False

    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """Called by PyAudio when mic data is available."""
        if self.running:
            self.send_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def audio_output_callback(self, in_data, frame_count, time_info, status):
        """Called by PyAudio when speaker needs data."""
        try:
            data = self.recv_queue.get_nowait()
            # Pad or trim to match frame_count
            expected_size = frame_count * 2  # 16-bit = 2 bytes
            if len(data) < expected_size:
                data += b'\x00' * (expected_size - len(data))
            elif len(data) > expected_size:
                data = data[:expected_size]
            return (data, pyaudio.paContinue)
        except queue.Empty:
            return (b'\x00' * frame_count * 2, pyaudio.paContinue)

    async def run(self):
        """Main loop - connect to API and stream audio."""
        print("Connecting to PersonaPlex API...")

        uri = f"{API_URL}?token={API_TOKEN}"

        async with websockets.connect(uri, open_timeout=120) as ws:
            # Wait for session start
            msg = json.loads(await ws.recv())
            session_id = msg['payload']['session_id']
            print(f"Session started: {session_id}")
            print("Speak into your microphone. Press Ctrl+C to stop.\n")

            # Start audio streams
            self.running = True

            input_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=FRAME_SIZE,
                stream_callback=self.audio_input_callback
            )

            output_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=FRAME_SIZE,
                stream_callback=self.audio_output_callback
            )

            input_stream.start_stream()
            output_stream.start_stream()

            try:
                # Create tasks for sending and receiving
                send_task = asyncio.create_task(self._send_loop(ws))
                recv_task = asyncio.create_task(self._recv_loop(ws))

                await asyncio.gather(send_task, recv_task)

            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                self.running = False
                input_stream.stop_stream()
                output_stream.stop_stream()
                input_stream.close()
                output_stream.close()

                await ws.send(json.dumps({"action": "stop"}))

    async def _send_loop(self, ws):
        """Send audio frames to the API."""
        while self.running:
            try:
                data = self.send_queue.get_nowait()
                await ws.send(data)
            except queue.Empty:
                await asyncio.sleep(0.01)

    async def _recv_loop(self, ws):
        """Receive audio from the API."""
        while self.running:
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                if isinstance(response, bytes):
                    print(f"Received audio: {len(response)} bytes")
                    self.recv_queue.put(response)
                else:
                    msg = json.loads(response)
                    if msg.get("event") == "error":
                        print(f"Error: {msg['payload']}")
            except asyncio.TimeoutError:
                pass
            except websockets.ConnectionClosed:
                break

    def cleanup(self):
        self.audio.terminate()


def main():
    client = RealtimeClient()
    try:
        asyncio.run(client.run())
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()
