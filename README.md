# 🎤 Speech-to-Speech Agent
### Gemini Flash (Native Audio I/O) + LangGraph + LangChain

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph State Machine                    │
│                                                                 │
│  START → [record] → [process] → [play] → [ask_continue] ──┐   │
│                         │                       │           │   │
│                    Gemini Flash            continue?        │   │
│                  (native audio I/O)         yes → loop      │   │
│                         │                   no  → END       │   │
│                    Tool Calls?                               │   │
│                  ┌──────┴──────┐                            │   │
│             get_time    calculate   search_kb               │   │
└─────────────────────────────────────────────────────────────────┘

Audio Flow (NO STT/TTS pipeline):
  Microphone PCM → Gemini Flash → Speaker PCM
       ↑ raw audio                 ↑ raw audio
  (no transcription)         (no text-to-speech)
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **Gemini Flash native audio** | Model understands & generates audio directly — no intermediate text |
| **LangGraph StateGraph** | Clean state machine for turn-taking, easy to extend |
| **LangChain tools** | Familiar tool interface; auto-converted to Gemini function declarations |
| **Files API upload** | Gemini requires audio via Files API for multimodal input |
| **MemorySaver** | Persists conversation across turns within a session |

---

## Setup

```bash
# 1. Install system dependency (PortAudio)
sudo apt install portaudio19-dev   # Ubuntu/Debian
brew install portaudio              # macOS

# 2. Install Python packages
pip install -r requirements.txt

# 3. Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# 4. Run
python speech_to_speech_agent.py
```

---

## Built-in Tools

| Tool | What it does |
|---|---|
| `get_current_time` | Returns current date & time |
| `calculate` | Evaluates math expressions |
| `search_knowledge_base` | Simple topic lookup (Python, LangGraph, AI…) |

Add your own tools by defining a `@tool` function and adding it to `TOOLS`.

---

## Extending

**Add a tool:**
```python
@tool
def my_tool(query: str) -> str:
    """Describe what the tool does."""
    return "result"

TOOLS = [get_current_time, calculate, search_knowledge_base, my_tool]
```

**Change voice:**
```python
voice_name="Aoede"   # options: Aoede, Charon, Fenrir, Kore, Puck
```

**Change recording duration:**
```python
RECORD_SECONDS = 8   # seconds
```
