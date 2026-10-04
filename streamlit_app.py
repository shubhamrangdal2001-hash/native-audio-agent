import streamlit as st
import asyncio
import threading
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Import the core agent logic
from speech_to_speech_agent import LiveAgent, GEMINI_MODEL

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gemini Speech-to-Speech",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
    }
    .ai-core {
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, #00d2ff 0%, #3a7bd5 70%, transparent 100%);
        border-radius: 50%;
        margin: 0 auto;
        animation: pulse 2s infinite ease-in-out;
        box-shadow: 0 0 50px rgba(0, 210, 255, 0.4);
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #00d2ff !important;
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'agent_running' not in st.session_state:
    st.session_state.agent_running = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'thread' not in st.session_state:
    st.session_state.thread = None

def add_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {msg}")
    if len(st.session_state.logs) > 50:
        st.session_state.logs.pop(0)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Settings")
    api_key = st.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    model_name = st.selectbox("Model", [GEMINI_MODEL, "gemini-3.1-flash-live-preview", "gemini-2.0-flash"])
    st.divider()
    st.info("This dashboard controls the local Speech-to-Speech agent. Make sure your microphone and speakers are connected.")

# Main UI
st.title("🎙️ Gemini Speech-to-Speech")
st.subheader("Native Multimodal AI Assistant")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    st.markdown('### AI Core Status')
    if st.session_state.agent_running:
        st.markdown('<div class="ai-core"></div>', unsafe_allow_html=True)
        st.success("AI is Active")
    else:
        st.markdown('<div class="ai-core" style="background: gray; box-shadow: none; animation: none; opacity: 0.3;"></div>', unsafe_allow_html=True)
        st.warning("AI is Offline")
    
    st.divider()
    
    if not st.session_state.agent_running:
        if st.button("🚀 Start Agent", use_container_width=True):
            if not api_key:
                st.error("Please enter an API Key first.")
            else:
                st.session_state.agent_running = True
                add_log("Initializing Live Agent...")
                # Start agent in a background thread
                def run_agent():
                    agent = LiveAgent(api_key)
                    # Note: In a real app, we might want to capture logs from the agent 
                    # and push them to st.session_state. This is a simplified version.
                    asyncio.run(agent.run())

                st.session_state.thread = threading.Thread(target=run_agent, daemon=True)
                st.session_state.thread.start()
                add_log("Agent started in background.")
                st.rerun()
    else:
        if st.button("🛑 Stop Agent", use_container_width=True):
            st.session_state.agent_running = False
            add_log("Stopping Agent...")
            # In a real app, we would signal the thread to stop gracefully.
            # Here we just stop tracking it in UI.
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="status-card" style="height: 500px; overflow-y: auto;">', unsafe_allow_html=True)
    st.markdown('### System Logs')
    for log in reversed(st.session_state.logs):
        st.text(log)
    st.markdown('</div>', unsafe_allow_html=True)

    # Memory Vault Section
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    st.markdown('### 🧠 Memory Vault')
    if os.path.exists("memory.json"):
        import json
        with open("memory.json", "r") as f:
            memory_data = json.load(f)
        
        # Display Facts
        facts = {k: v for k, v in memory_data.items() if k != "history"}
        if facts:
            st.markdown("#### 📌 Remembered Facts")
            for k, v in facts.items():
                st.markdown(f"**{k.capitalize()}**: {v}")
        
        # Display History
        if "history" in memory_data and memory_data["history"]:
            st.markdown("#### 📜 Recent Conversation")
            for item in reversed(memory_data["history"]):
                with st.container():
                    st.markdown(f"**{item['timestamp']}**")
                    with st.chat_message("assistant"):
                        st.write(item['ai'])
                    st.divider()
        
        if not facts and not ("history" in memory_data and memory_data["history"]):
            st.info("Memory is empty.")
    else:
        st.info("No memory file found.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.5);'>"
    "Built with ❤️ using Streamlit & Gemini 2.5"
    "</div>",
    unsafe_allow_html=True
)
