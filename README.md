# 🎤 Chica Voice Assistant with Animated Avatar

A local voice assistant with animated avatar that uses local models for natural language processing, speech synthesis, and speech recognition. Created by Claudinei Villada with help from Vibe Code.

## ✨ Features

- **🎤 Voice Recognition**: Real-time speech-to-text using Whisper (local)
- **🤖 Dual LLM Support**: Works with **Ollama** (default) or **LM Studio**
- **🗣️ Dual TTS System**:
  - **Kokoro-TTS** (default) — Natural Portuguese voice (pf_dora)
  - **Qwen3-TTS** — Advanced system with voice Serena
- **👤 Animated Avatar**: Real-time avatar animation synchronized with speech (Pygame)
- **🔊 Wake Word Detection**: Multiple wake words for activation ("olá chica", "ei chica", etc.)
- **🖥️ Local Commands**: Execute system commands with voice confirmation ("abra o navegador")
- **🧠 Persistent Memory**: Remembers important facts between conversations
- **⚡ Real-time Processing**: Low-latency adaptive noise detection
- **📁 Modular Architecture**: Clean separation of concerns across 9 modules
- **🔄 Automatic Fallback**: Kokoro-TTS fallback if Qwen3-TTS fails
- **🔇 Voice Interruption**: Say "calado" or "silêncio" to stop speech

## 📋 System Requirements

1. 🐍 **Python 3.8+**
2. 💾 **8GB RAM** (16GB recommended)
3. 🎤 Working microphone
4. 🔊 Working speakers
5. 🌐 Internet connection (for initial model downloads)
6. 🖥️ Graphical system (for avatar — optional)

## ⚙️ Installation

### 1. Install Python 3.8+ if needed:
- **Windows**: https://www.python.org/downloads/
- **macOS**: `brew install python`
- **Linux**: `sudo apt install python3 python3-pip`

### 2. Create virtual environment:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

### 3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 4. Install system dependencies for Pygame (optional — avatar only):

**macOS:**
```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev portaudio19-dev sox
```

## 🔧 LLM Provider (Choose One)

### Option A — Ollama (default)
1. Download: https://ollama.ai/
2. Start: `ollama serve`
3. Pull a model: `ollama pull qwen3:1.7b`

### Option B — LM Studio (alternative)
1. Download: https://lmstudio.ai/
2. Load a model in the UI
3. Enable API: Settings > Enable API (port 1234)
4. Install: `pip install openai`
5. Edit `config.py`:
   ```python
   LLM_PROVIDER = 'lm_studio'
   LLM_MODEL = 'bonsai-8b'   # or your loaded model
   ```

## 🎯 Configuration

All settings are centralized in `config.py`:

```python
# LLM Provider
LLM_PROVIDER = 'lm_studio'        # 'ollama' or 'lm_studio'
LLM_MODEL = 'bonsai-8b'
LLM_TEMPERATURE = 0.7
LLM_NUM_PREDICT = 300

# Assistant
ASSISTANT_NAME = "Chica"
ASSISTANT_GREETING = " Eu sou a Chica, sua assistente. Como posso ajudar?"

# Avatar (optional — saves memory)
AVATAR_ENABLE = True               # Set False to disable

# Wake words
WAKE_WORDS = ["olá chica", "ei chica", "chica", ...]

# TTS Voice
TTS_VOICE = 'pf_dora'             # Portuguese voice for Kokoro
TTS_SYSTEM = 'kokoro'             # 'kokoro' or 'qwen3'
TTS_SAMPLE_RATE = 24000

# Microphone sensitivity
SPEECH_THRESHOLD = 0.005
SILENCE_DURATION = 1.5
INACTIVITY_TIMEOUT = 15.0

# Speech-to-Text (Whisper)
WHISPER_MODEL = 'base'             # 'tiny', 'base', 'small', 'medium', 'large'
```

## 🚀 Running

### 1. Start your LLM provider:
- **Ollama**: `ollama serve`
- **LM Studio**: Open app with API enabled

### 2. Run the assistant:
```bash
python app.py
```

### 3. At startup, choose TTS system:
- `1` = Kokoro-TTS (default, Portuguese)
- `2` = Qwen3-TTS (voice Serena)
- `3` = Use config.py setting

## 🗣️ Voice Commands

| Command | Action |
|---------|--------|
| "olá chica" / "ei chica" | Wake up the assistant |
| "qual é a capital do Brasil?" | Normal conversation |
| "abra o navegador" | Open browser (asks confirmation) |
| "abra o Spotify" | Open Spotify |
| "abra o VS Code" | Open Visual Studio Code |
| "abra o terminal" | Open terminal |
| "abra o Discord" | Open Discord |
| "calado" / "silêncio" | Interrupt speech |

All local commands require voice confirmation. Say "sim" to confirm, "não" to cancel.

## 🧠 Persistent Memory

Chica remembers important information between conversations:
- Automatically extracts facts (name, preferences, personal info)
- Stored in `chica_memory.json` (max 10 entries)
- Expires after 30 days
- Injected into context — no performance impact
- Extracted every 5 interactions via LLM

## 📁 Project Structure

```
kokoro/
├── app.py              (~1650 lines)  — Main entry point
├── config.py           (~ 242 lines)  — Centralized configuration
├── llm_client.py       (~ 284 lines)  — Ollama / LM Studio client
├── avatar.py           (~ 252 lines)  — Animated Pygame avatar
├── tts_engine.py       (~ 317 lines)  — Text-to-speech (Kokoro / Qwen3)
├── audio_detector.py   (~ 344 lines)  — Voice capture and detection
├── commands.py         (~ 177 lines)  — Local OS command execution
├── memory_manager.py   (~ 186 lines)  — Persistent memory
├── log.py              (~ 109 lines)  — Colored structured logging
├── chica_img/          ─ Avatar images (PNG)
├── chica_memory.json   ─ Saved memories (auto-generated)
├── requirements.txt    ─ Python dependencies
├── leiame.txt          ─ Documentation (Portuguese)
└── README.md           ─ This file
```

## 🔄 System Flow

```
🎤 Microphone → Whisper (STT) → LLM (Ollama/LM Studio)
   → TTS (Kokoro/Qwen3) → 🔊 Speaker + 👤 Avatar
```

## 🎯 Usage Tips

1. Speak clearly in a quiet environment
2. Wait for the audio response before speaking again
3. Say "calado" to interrupt speech at any time
4. Avatar helps visualize when the assistant is speaking
5. Avatar can be disabled with `AVATAR_ENABLE = False` to save memory

## 🍓 Raspberry Pi 4

Recommendations for running on Raspberry Pi 4:
- Use Whisper `tiny`: `WHISPER_MODEL = 'tiny'`
- Disable avatar: `AVATAR_ENABLE = False`
- Use Ollama with small models (qwen3:0.6b, llama3.2:1b)
- Disable thinking: `THINKING_ENABLED = False`
- Use LM Studio with models under 3B parameters

## 🤝 Contributing

To report issues or suggest improvements:
1. Check if the issue has already been reported
2. Include system information
3. Describe steps to reproduce
4. Include full error messages
