# 🎤 Chica Voice Assistant with Animated Avatar

A local voice assistant with animated avatar that uses local models for natural language processing, speech synthesis, and speech recognition. Created by Claudinei Villada with help from Vibe Code.

## ✨ Features

- **🎤 Voice Recognition**: Real-time speech-to-text using Whisper
- **🤖 Local LLM**: Language processing using Ollama with local models
- **🗣️ Dual TTS System**: Natural-sounding speech synthesis with choice between:
  - **Kokoro-TTS** (default) - Original system
  - **Qwen3-TTS** - Advanced system with voice Serena in Portuguese
- **👤 Animated Avatar**: Real-time avatar animation synchronized with speech
- **🔊 Wake Word Detection**: Multiple wake words for activation
- **⚡ Real-time Processing**: Low-latency audio processing
- **📁 Modular Configuration**: Centralized configuration in `config.py`
- **🔄 Automatic Fallback**: Automatic fallback to Kokoro-TTS if Qwen3-TTS fails

## 📋 System Requirements

1. 🐍 **Python 3.8** or higher
2. 💾 **8GB RAM** minimum (16GB recommended)
3. 🎤 Working microphone
4. 🔊 Working speakers
5. 🌐 Internet connection (for downloading models)
6. 🖥️ Graphical system (for avatar display)

## ⚙️ Installation

### 1. Install Python 3.8+ if not already installed:
- **Windows**: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- **macOS**: `brew install python`
- **Linux**: `sudo apt install python3 python3-pip`

### 2. Create a virtual environment (recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

**Note about Qwen3-TTS**: The `qwen3-tts` package may require additional setup. If you encounter installation issues:
```bash
# Try installing with specific version
pip install qwen3-tts>=0.1.0

# Or install from GitHub (if pip version fails)
pip install git+https://github.com/QwenLM/Qwen-TTS.git
```

### 4. Install system dependencies for Pygame (avatar):

**macOS (Intel and Apple Silicon M1/M2/M3/M4):**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install SDL2 and audio dependencies for Pygame
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf portaudio

# Install SoX for Qwen3-TTS (required for audio processing)
brew install sox

# For Apple Silicon Macs, ensure Python is properly linked
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev portaudio19-dev sox
```

**Windows:**
- Download Pygame and SDL2 installers from official website
- Install SoX from: http://sox.sourceforge.net/

## 🔧 Ollama Installation (Required)

Ollama is required for the local language model.

### 1. Download and install Ollama:
- Official site: [https://ollama.ai/](https://ollama.ai/)
- Or use the commands below:

```bash
# macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows (PowerShell as administrator)
winget install Ollama.Ollama
```

### 2. Start the Ollama service:
```bash
# In a separate terminal
ollama serve
```

### 3. Download a model (recommended):
```bash
# Recommended Portuguese model
ollama pull llama3.2:3b

# Alternative models
ollama pull mistral:7b
ollama pull gemma:2b
```

## 🎯 Configuration

### 1. All configurations are centralized in `config.py`:

```python
# General settings
ASSISTANT_NAME = "Chica"
ASSISTANT_GREETING = " Eu sou a Chica, sua assistente. Como posso ajudar?"

# Avatar settings
AVATAR_ENABLE = True  # True: enables avatar window, False: completely disables avatar
AVATAR_WINDOW_SIZE_CM = 7  # Window size in centimeters
AVATAR_IMAGE_DIR = "chica_img"  # Folder with avatar images
AVATAR_BLINK_INTERVAL = 3.0  # Seconds between blinks
AVATAR_SPEAK_ANIMATION_SPEED = 0.1  # Speech animation speed

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
AUDIO_DEVICE = "Isolamento de Voz"  # Audio device name

# Wake words
WAKE_WORDS = [
    "olá chica", "ei chica", "chica", "ok chica",
    "shika", "shica", "hei chica", "hei shica"
]

# Model settings
OLLAMA_MODEL = 'llama3.2:3b'
WHISPER_MODEL = 'base'
TTS_VOICE = 'pf_dora 80% mais if_sara 20%'

# ============================================================================
# TTS SYSTEM CONFIGURATION (KOKORO vs QWEN3)
# ============================================================================
# TTS system to use
# Options: 'kokoro' (default) or 'qwen3'
TTS_SYSTEM = 'kokoro'

# Qwen3-TTS specific settings
QWEN3_MODEL = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'  # Qwen3-TTS model
QWEN3_VOICE = 'serena'  # Voice for Portuguese (serena)
QWEN3_LANGUAGE = 'portuguese'  # Language for synthesis
```

### 2. Test your microphone:
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### 3. Verify avatar images:
- Ensure the `chica_img/` folder contains:
  - `chica_normal.png` (initial state)
  - `chica_olho.png` (blinking)
  - `chica_boca.png` (speaking)
  - `chica_olho_boca.png` (blinking while speaking)

## 🚀 Running the Program

### 1. Make sure Ollama is running:
```bash
ollama serve
```

### 2. Run the assistant with avatar:
```bash
python app.py
```

### 3. Voice commands:
- **Wake words** - To activate the assistant:
  - "olá chica", "ei chica", "chica", "ok chica"
  - "shika", "shica", "hei chica", "hei shica"
  - Also accepts: "ola chica" (without accent), "oi chica", "hey chica", etc.
- **"pare"** or **"para"** - To interrupt speech
- **15 seconds of silence** - Goes back to sleep

### 4. Avatar interface:
- 7x7cm window with animations synchronized with speech
- Natural blinking when not speaking
- Speech animation alternating between open and closed mouth
- **Enable/Disable control**: Use `AVATAR_ENABLE = False` in `config.py` to completely disable the avatar (useful for systems without graphical interface or to save resources)

## 🎛️ Advanced Configuration

### 1. To use GPU (if available):
```bash
# Uninstall current PyTorch
pip uninstall torch

# Install PyTorch with CUDA (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for Apple Silicon
pip install torch torchvision torchaudio
```

### 2. Voice settings:
- Single voice: `'pf_dora'`
- Mixed voice: `'pf_dora 80% mais if_sara 20%'`
- Other voices available in Kokoro

### 3. Thinking processing settings:
```python
# ============================================================================
# THINKING CONFIGURATION (MODEL PROCESSING)
# ============================================================================

THINKING_ENABLED = True              # Enable/disable model thinking mode True or False
THINKING_TIMEOUT = 30                # Timeout in seconds for thinking processing
```
- `THINKING_ENABLED`: Controls whether the system processes the model's `thinking` field
- `THINKING_TIMEOUT`: Maximum time to process complex responses

### 4. Avatar size:
- Modify `AVATAR_WINDOW_SIZE_CM` in `config.py`
- Size is automatically converted to pixels

## 🎤 TTS System Selection

The assistant now supports two TTS (Text-to-Speech) systems that you can choose from:

### 1. **Interactive Selection at Startup**
When you run the assistant, you'll see this selection menu:
```
🎤 SELECIONE O SISTEMA TTS:
   1. Kokoro-TTS (padrão)
   2. Qwen3-TTS (voz Serena em português)
   3. Usar configuração atual do config.py
Escolha (1/2/3) [3]:
```

### 2. **TTS System Features**

#### **Kokoro-TTS (Default)**
- ✅ **Stable and reliable**
- ✅ **Multiple voice options** (configured in `TTS_VOICE`)
- ✅ **No external dependencies** (works out of the box)
- ✅ **Fast synthesis**

#### **Qwen3-TTS (Advanced)**
- ✅ **High-quality voice synthesis** with voice "Serena" in Portuguese
- ✅ **Natural-sounding speech** with better prosody
- ✅ **CustomVoice model** with multiple voice options
- ⚠️ **Requires SoX** (Sound eXchange) installed on your system
- ⚠️ **Larger model size** (~0.6B parameters)

### 3. **Automatic Fallback System**
- If Qwen3-TTS fails to initialize (missing dependencies, import errors, etc.)
- The system automatically falls back to Kokoro-TTS
- You'll see a warning message but the assistant continues working

### 4. **Configuration Options**
In `config.py`, you can set:
- `TTS_SYSTEM`: Default system ('kokoro' or 'qwen3')
- `QWEN3_MODEL`: Qwen3-TTS model name
- `QWEN3_VOICE`: Voice for Qwen3-TTS ('serena' for Portuguese)
- `QWEN3_LANGUAGE`: Language for synthesis ('portuguese')

### 5. **Troubleshooting Qwen3-TTS**
If you see "SoX could not be found!" error:
```bash
# macOS (Intel and Apple Silicon)
brew install sox

# Ubuntu/Debian
sudo apt-get install sox

# Windows
# Download from: http://sox.sourceforge.net/
```

Note: Qwen3-TTS will work with a warning even if SoX is not installed, but audio quality may be affected.

## 🔧 System Architecture

### 📁 File structure:
```
kokoro/
├── app.py              # Main program with avatar
├── config.py            # Centralized configuration
├── requirements.txt     # Python dependencies
├── leiame.txt          # Portuguese manual
├── README.md           # English documentation
└── chica_img/          # Avatar images
    ├── chica_normal.png
    ├── chica_olho.png
    ├── chica_boca.png
    └── chica_olho_boca.png
```

### 🔄 System flow:
1. Wake word detection → 2. STT (Whisper) → 3. LLM (Ollama) → 4. TTS (Kokoro/Qwen3) → 5. Avatar animation

### 🎤 TTS System Integration:
- **Dual TTS Support**: The system can use either Kokoro-TTS or Qwen3-TTS
- **Interactive Selection**: Choose TTS system at startup
- **Automatic Fallback**: If Qwen3-TTS fails, automatically switches to Kokoro-TTS
- **Configurable**: All TTS settings in `config.py`

## 🛠️ Troubleshooting Common Issues

### 1. **Qwen3-TTS Installation Problems**
```bash
# If pip install qwen3-tts fails:
pip install git+https://github.com/QwenLM/Qwen-TTS.git

# Or try installing dependencies manually:
pip install torch torchaudio transformers
pip install qwen3-tts>=0.1.0
```

### 2. **SoX Not Found Error (Qwen3-TTS)**
```bash
# macOS
brew install sox

# Ubuntu/Debian
sudo apt-get install sox

# Windows
# Download from: http://sox.sourceforge.net/
```

### 3. **Pygame/SDL2 Issues on macOS**
```bash
# Reinstall SDL2 dependencies
brew reinstall sdl2 sdl2_image sdl2_mixer sdl2_ttf portaudio

# Reinstall Pygame
pip uninstall pygame
pip install pygame>=2.6.0
```

### 4. **Audio Device Issues**
```python
# Check available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Update AUDIO_DEVICE in config.py with correct device name
```

### 5. **Ollama Connection Issues**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is downloaded
ollama list

# Pull the model if not present
ollama pull llama3.2:3b
```

## 🎯 Usage Tips

1. Speak clearly in a quiet environment
2. Wait for the "beep" before speaking
3. Use complete sentences for better responses
4. The avatar helps visualize when the assistant is speaking
5. To interrupt, say "pare" or "para"

## 🤖 Supported Models and Thinking Processing

The system supports various Ollama models with intelligent processing:

### 1. **Models with thinking (internal reasoning):**
   - `qwen3:0.6b` - Uses `thinking` field for complex reasoning
   - Other models that implement the thinking pattern

### 2. **Models without thinking:**
   - `llama3.2:3b` - Direct responses in `content` field
   - `mistral:7b` - Direct responses
   - `gemma:2b` - Direct responses

### 3. **Intelligent thinking processing:**
   - ✅ **Response extraction**: Extracts final answer from `thinking` field
   - ✅ **Repetition detection**: Avoids repeating previous answers
   - ✅ **Reasoning filtering**: Removes internal reasoning lines
   - ✅ **Intelligent fallback**: Multiple extraction strategies

### 4. **Thinking configuration:**
   - Enable/disable with `THINKING_ENABLED` in `config.py`
   - Adjust timeout with `THINKING_TIMEOUT` (default: 30 seconds)

## 📊 Performance Analysis and Recommendations

This section provides detailed performance analysis and optimization recommendations based on extensive testing of the dual TTS system.

### 🔍 **Performance Test Results**

#### **Qwen3-TTS Performance on Apple Silicon (MPS)**
| Metric | Result | Notes |
|--------|--------|-------|
| **Model Loading Time** | 4.5-5.0 seconds | With MPS acceleration enabled |
| **Average Synthesis Time** | 5-8 seconds per phrase | For phrases of 10-20 words |
| **Streaming vs Non-Streaming** | Non-streaming mode is 40% faster | Use `non_streaming_mode=True` |
| **Impact of torch.compile** | 10x slower on MPS | Disabled by default in config |
| **Cache Effectiveness** | ~90% hit rate for frequent short phrases | For phrases ≤ 50 characters |

#### **Comparative Performance (Kokoro-TTS vs Qwen3-TTS)**
| TTS System | Speed | Quality | Best Use Case |
|------------|-------|---------|---------------|
| **Kokoro-TTS** | ⚡ Very Fast (0.5-1s) | Good | Real-time responses, general use |
| **Qwen3-TTS** | 🐢 Slow (5-8s) | Excellent | High-quality Portuguese with Serena voice |

### ⚙️ **Optimization Recommendations**

#### **1. MPS Acceleration (Apple Silicon)**
- ✅ **Already optimized**: Qwen3-TTS now uses MPS (GPU) by default on Apple Silicon Macs
- ⚠️ **Performance limit**: Even with MPS, Qwen3-TTS has inherent latency due to model architecture
- 🔧 **Verification**: Check terminal output for "Dispositivo TTS: mps" to confirm MPS is active

#### **2. Configuration Settings**
```python
# config.py - Optimal settings for Qwen3-TTS
QWEN3_USE_COMPILE = False  # torch.compile degrades performance on MPS
TTS_SYSTEM = 'kokoro'      # Default for real-time use, change to 'qwen3' when needed
```

#### **3. Cache System**
- **Automatic caching**: Frequently used short phrases (≤50 chars) are cached
- **Cache size**: LRU cache with 32 entries
- **Performance gain**: 90%+ reduction in synthesis time for cached phrases

#### **4. Warnings Suppression**
- Transformer/Qwen3-TTS warnings are suppressed for cleaner output
- Critical errors still appear
- The "Setting `pad_token_id` to `eos_token_id`:2150" warning is harmless and suppressed

### 🚀 **Installation Recommendations**

#### **For Apple Silicon (M1/M2/M3/M4) Users**
1. **Ensure PyTorch MPS support**:
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
   ```

2. **Install SoX for optimal Qwen3-TTS audio quality**:
   ```bash
   brew install sox
   ```

3. **Install Pygame dependencies**:
   ```bash
   brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf portaudio
   ```

#### **For NVIDIA GPU Users**
1. **Install CUDA-enabled PyTorch**:
   ```bash
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Consider flash-attn for acceleration** (CUDA only):
   ```bash
   pip install flash-attn
   ```

### 🎯 **Usage Recommendations**

#### **Recommended Configuration**
- **Primary TTS**: Kokoro-TTS (`TTS_SYSTEM = 'kokoro'`)
- **Use Qwen3-TTS when**: You need high-quality Portuguese with Serena voice
- **Switch at runtime**: Use the interactive menu at startup (option 2 for Qwen3-TTS)

#### **Performance Tips**
1. **For real-time conversations**: Stick with Kokoro-TTS
2. **For high-quality responses**: Switch to Qwen3-TTS for specific queries
3. **Use short phrases**: Qwen3-TTS performs better with concise text
4. **Allow warm-up**: First Qwen3-TTS response will be slowest (4-5s loading + 5-8s synthesis)

#### **Troubleshooting Performance Issues**
| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Qwen3-TTS very slow (>10s) | Running on CPU instead of MPS | Check "Dispositivo TTS: mps" in terminal |
| Terminal flooded with warnings | Warning suppression not working | Update app.py with latest warning filters |
| No audio from Qwen3-TTS | SoX not installed | `brew install sox` (macOS) or `sudo apt install sox` (Linux) |
| torch.compile makes things worse | MPS incompatibility | Ensure `QWEN3_USE_COMPILE = False` |

### 📈 **Expected Performance**
- **Kokoro-TTS**: Near real-time (0.5-1 second response time)
- **Qwen3-TTS**: 5-8 seconds for first response, 4-7 seconds for subsequent responses
- **Cache hits**: < 1 second for frequent short phrases

### 🔮 **Future Optimization Potential**
1. **Model quantization**: 8-bit quantization could reduce model size but currently incompatible with Qwen3-TTS architecture
2. **Batch processing**: Processing multiple phrases simultaneously
3. **Model pruning**: Using smaller variant if available (0.3B or 0.1B)
4. **Hardware acceleration**: Future PyTorch/MPS optimizations may improve performance

## 🤝 Contributing

To report issues or suggest improvements:
1. Check if the issue has already been reported
2. Include system information
3. Describe steps to reproduce the issue
4. Include complete error messages

## 📞 Support

For additional help:
- Consult this manual
- Check error messages in the terminal
- Test components individually
- Consult library documentation

## 📄 License

This project is for educational and personal use.

## 🙏 Acknowledgments

- **Ollama** for local LLM capabilities
- **OpenAI Whisper** for speech recognition
- **Kokoro** for speech synthesis
- **Pygame** for avatar animation
- **Vibe Code** for development assistance