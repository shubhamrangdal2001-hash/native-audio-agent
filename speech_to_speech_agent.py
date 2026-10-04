"""
Speech-to-Speech Agent using Gemini Multimodal Live API
=========================================================
- NO STT → Text → TTS pipeline
- Uses Gemini's native multimodal audio understanding & generation
- Powered by the Live API (WebSocket) for low latency
"""

import os
import asyncio
import pyaudio
import base64
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import time

load_dotenv()

from google import genai
from google.genai import types

# ── Configuration ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Using the confirmed working model from 2026
GEMINI_MODEL   = "gemini-3.1-flash-live-preview"

FORMAT         = pyaudio.paInt16
CHANNELS       = 1
RATE           = 16000  # Input rate for Gemini
CHUNK          = 512    # Smaller chunks for lower latency

SYSTEM_PROMPT = """
You are a helpful, friendly, and concise voice assistant.
You receive audio directly and respond with audio.
Keep responses short and natural for spoken conversation.

MEMORY CAPABILITIES:
- You have access to a long-term memory. 
- If the user tells you something personal or a preference, use the `save_to_memory` tool.
- If you need to recall something, use the `search_memory` tool.
- Always be warm, clear, and easy to understand in spoken form.
"""

# ── Memory Management ────────────────────────────────────────────────────────
class MemoryManager:
    def __init__(self, file_path="memory.json"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump({}, f)

    def save(self, key: str, value: str):
        with open(self.file_path, "r") as f:
            data = json.load(f)
        data[key.lower()] = value
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)
        return f"Successfully remembered: {key} is {value}"

    def log_interaction(self, user_text: str, ai_text: str):
        with open(self.file_path, "r") as f:
            data = json.load(f)
        if "history" not in data:
            data["history"] = []
        data["history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_text,
            "ai": ai_text
        })
        # Keep last 20 interactions in history
        data["history"] = data["history"][-20:]
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)

    def search(self, query: str):
        with open(self.file_path, "r") as f:
            data = json.load(f)
        results = [f"{k}: {v}" for k, v in data.items() if query.lower() in k.lower() or query.lower() in v.lower()]
        if not results:
            return "No matching information found in memory."
        return "Found in memory: " + " | ".join(results)

memory_manager = MemoryManager()

# ── Tools ────────────────────────────────────────────────────────────────────
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("It is %A, %B %d %Y, %I:%M %p")

def calculate(expression: str) -> str:
    """Safely evaluates basic arithmetic."""
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "I can only evaluate basic arithmetic expressions."
    try:
        return f"The result is {eval(expression, {'__builtins__': {}})}"
    except Exception as e:
        return f"Error: {e}"

def save_to_memory(info_key: str, info_value: str) -> str:
    """
    Saves a fact or preference about the user to long-term memory.
    Args:
        info_key: The topic (e.g. 'user_name', 'favorite_color')
        info_value: The detail to remember
    """
    return memory_manager.save(info_key, info_value)

def search_memory(query: str) -> str:
    """
    Searches the long-term memory for information.
    Args:
        query: The keyword to search for
    """
    return memory_manager.search(query)

TOOLS = [get_current_time, calculate, save_to_memory, search_memory]

# ── Audio Interface ──────────────────────────────────────────────────────────
class AudioInterface:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.in_stream = None
        self.out_stream = None

    def start(self):
        # Re-initialize PyAudio if it was terminated
        try:
            self.pa.get_device_count()
        except:
            self.pa = pyaudio.PyAudio()

        # List devices for debugging
        print("\n[Audio] Available devices:")
        for i in range(self.pa.get_device_count()):
            dev = self.pa.get_device_info_by_index(i)
            print(f"  {i}: {dev['name']} (In: {dev['maxInputChannels']}, Out: {dev['maxOutputChannels']})")
        
        self.in_stream = self.pa.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK
        )
        self.out_stream = self.pa.open(
            format=FORMAT, channels=CHANNELS, rate=24000, 
            output=True, frames_per_buffer=CHUNK
        )
        print("[Audio] Streams initialized.")

    def stop(self):
        """Stops the streams but keeps the PortAudio engine alive for reconnection."""
        if self.in_stream:
            try:
                self.in_stream.stop_stream()
                self.in_stream.close()
            except: pass
            self.in_stream = None
        if self.out_stream:
            try:
                self.out_stream.stop_stream()
                self.out_stream.close()
            except: pass
            self.out_stream = None

    def terminate(self):
        """Final cleanup of the PortAudio engine."""
        self.stop()
        self.pa.terminate()

    async def read_mic(self, queue: asyncio.Queue):
        """Reads from microphone and puts into queue."""
        while True:
            data = self.in_stream.read(CHUNK, exception_on_overflow=False)
            await queue.put(data)
            await asyncio.sleep(0)

    def play_audio(self, data: bytes):
        """Plays audio data to speaker."""
        try:
            if self.out_stream and self.out_stream.is_active():
                self.out_stream.write(data)
        except Exception as e:
            print(f"[Audio Error] Playback failed: {e}")

# ── Agent ────────────────────────────────────────────────────────────────────
class LiveAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        self.audio = AudioInterface()
        self.mic_queue = asyncio.Queue()

    async def run(self):
        self.audio.start()
        config = {
            'system_instruction': SYSTEM_PROMPT,
            'response_modalities': ['AUDIO'], 
            'speech_config': {
                'voice_config': {
                    'prebuilt_voice_config': {'voice_name': 'Puck'} 
                }
            }
        }

        print(f"\n[Live] Connecting to {GEMINI_MODEL}...")
        try:
            while True:
                # Reset audio and queue for each new attempt
                self.audio.stop() 
                self.audio.start()
                while not self.mic_queue.empty():
                    try: self.mic_queue.get_nowait()
                    except asyncio.QueueEmpty: break

                try:
                    async with self.client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
                        print("[OK] Connected! Talk now (Ctrl+C to quit)\n")
                        await asyncio.sleep(1) 
                        
                        self.mic_task = asyncio.create_task(self.audio.read_mic(self.mic_queue))
                        await self._process_io(session)
                except Exception as e:
                    print(f"\n[Error] Live session error: {e}")
                    print("[Live] Attempting to reconnect in 3 seconds...")
                    await asyncio.sleep(3)
                finally:
                    if hasattr(self, 'mic_task'):
                        self.mic_task.cancel()
                        try: await self.mic_task
                        except: pass
        finally:
            self.audio.terminate() # Use terminate only at the very end

    async def _process_io(self, session):
        """Handles both sending mic data and receiving model responses."""
        
        async def send_loop():
            chunks_sent = 0
            last_log = time.time()
            while True:
                try:
                    audio_data = await self.mic_queue.get()
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=audio_data, 
                            mime_type="audio/pcm;rate=16000"
                        )
                    )
                    chunks_sent += 1
                    if time.time() - last_log > 5:
                        print(f" [Mic] Active - sent {chunks_sent} chunks in last 5s")
                        chunks_sent = 0
                        last_log = time.time()
                except Exception as e:
                    print(f" [Mic Error] Send failed: {e}")
                    break

        async def receive_loop():
            current_ai_text = ""
            async for message in session.receive():
                if message.server_content:
                    model_turn = message.server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.inline_data:
                                self.audio.play_audio(part.inline_data.data)
                            if part.text:
                                print(f" [AI Text] {part.text}")
                                current_ai_text += part.text
                    
                    if message.server_content.turn_complete:
                        print(" [AI turn complete]")
                        # Automatically log the interaction
                        if current_ai_text:
                            memory_manager.log_interaction("Voice Input", current_ai_text)
                            current_ai_text = ""

                if message.tool_call:
                    for call in message.tool_call.function_calls:
                        print(f"[Tool] Calling {call.name}({call.args})")
                        # Handle tool call
                        if call.name == "get_current_time":
                            result = get_current_time()
                        elif call.name == "calculate":
                            result = calculate(call.args['expression'])
                        elif call.name == "save_to_memory":
                            result = save_to_memory(call.args['info_key'], call.args['info_value'])
                        elif call.name == "search_memory":
                            result = search_memory(call.args['query'])
                        else:
                            result = "Tool not found"
                        
                        print(f"       Result: {result}")
                        await session.send(
                            input=types.LiveClientToolResponse(
                                function_responses=[
                                    types.LiveClientFunctionResponse(
                                        name=call.name,
                                        id=call.id,
                                        response={'result': result}
                                    )
                                ]
                            )
                        )
                        print(f" [Tool Response Sent]")

        # Run both loops concurrently
        tasks = [asyncio.create_task(send_loop()), asyncio.create_task(receive_loop())]
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        return "DONE"

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found.")
    else:
        agent = LiveAgent(GEMINI_API_KEY)
        try:
            asyncio.run(agent.run())
        except KeyboardInterrupt:
            print("\n[Bye] Session closed.")
