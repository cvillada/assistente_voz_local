# 🎤 Chica — Assistente de Voz com Avatar Animado

Assistente de voz local com avatar animado, usando modelos locais para NLP, síntese e reconhecimento de fala. Criado por **Claudinei Villada**.

## ✨ Funcionalidades

- **🎤 Reconhecimento de Voz**: STT em tempo real com Whisper (modelo `turbo`).
  Usa **whisper original** (MPS/GPU) no Mac Apple Silicon e **faster-whisper** (CTranslate2, 4x mais rápido) em CPU/ARM.
  Detecção automática de hardware — não precisa configurar nada.
- **🤖 LLM Local**: Suporte a **LM Studio** (padrão), **Ollama** ou **llama.cpp**
- **🗣️ TTS Triplo**:
  - **Kokoro-TTS** (padrão) — Voz natural em português (pf_dora)
  - **Edge-TTS** — Síntese neural online (voz Thalita, pt-BR), com fallback automático para Kokoro se ficar offline
  - **Qwen3-TTS** — Sistema avançado com voz Serena
- **👤 Avatar Animado**: Animação em tempo real sincronizada com a fala (Pygame)
- **🔊 Wake Word**: Múltiplas palavras de ativação ("olá chica", "ei chica", etc.)
- **🖥️ Comandos Locais**: Executa comandos do sistema com confirmação por voz ("abra o navegador")
- **🔍 Pesquisa na Web**: Detecta automaticamente quando pesquisar na internet
- **📊 Informações do Sistema**: Consultas sobre disco, RAM, CPU, bateria, IP, etc.
- **🧠 Memória Persistente v2**: Lembra fatos importantes entre conversas (2 arquivos markdown)
- **⚡ Processamento em Tempo Real**: Detecção adaptativa de ruído com baixa latência
- **📁 Arquitetura Modular**: Separação clara em 12 módulos
- **🔄 Fallback Automático**: Kokoro-TTS como fallback se Edge-TTS ou Qwen3 falharem
- **🔇 Interrupção por Voz**: Diga "calado" ou "silêncio" para interromper a fala

## 📋 Requisitos de Sistema

1. 🐍 **Python 3.8+**
2. 💾 **8GB RAM** (16GB recomendado)
3. 🎤 Microfone funcionando
4. 🔊 Alto-falantes funcionando
5. 🌐 Internet (para download inicial de modelos; Edge-TTS requer internet)
6. 🖥️ Sistema gráfico (para o avatar — opcional)

## ⚙️ Instalação

### 1. Instale o Python 3.8+ se necessário:
- **Windows**: https://www.python.org/downloads/
- **macOS**: `brew install python`
- **Linux**: `sudo apt install python3 python3-pip`

### 2. Crie um ambiente virtual:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

### 3. Instale as dependências Python:
```bash
pip install -r requirements.txt
```

### 4. Dependências de sistema para Pygame (opcional — apenas avatar):
**macOS:**
```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev portaudio19-dev sox
```

## 🔧 Provedor LLM (Escolha Um)

### Opção A — LM Studio (padrão)
1. Baixe: https://lmstudio.ai/
2. Carregue um modelo na interface (ex: `bonsai-4b`)
3. Habilite a API: Settings > Enable API (porta 1234)
4. Instale: `pip install openai`

### Opção B — Ollama (alternativa)
1. Baixe: https://ollama.ai/
2. Inicie: `ollama serve`
3. Baixe um modelo: `ollama pull qwen3:1.7b`

### Opção C — llama.cpp (recomendado para SBC/Raspberry Pi)
1. Compile ou instale: https://github.com/ggml-org/llama.cpp
2. Baixe um GGUF (ex: `qwen2.5-3b-instruct-q4_k_m.gguf`)
3. Inicie o servidor:
   ```bash
   llama-server -m modelo.q4_k_m.gguf -c 4096 --port 8080
   ```
4. Instale: `pip install openai`
5. No `config.py`: `LLM_PROVIDER = 'llamacpp'`

## 🎯 Configuração

Todas as configurações estão centralizadas em `config.py`:

```python
# Provedor LLM
LLM_PROVIDER = 'lm_studio'        # 'lm_studio', 'ollama' ou 'llamacpp'
LLM_MODEL = 'bonsai-4b'
LLM_TEMPERATURE = 0.7
LLM_NUM_PREDICT = 300

# Assistente
ASSISTANT_NAME = "Chica"
ASSISTANT_GREETING = " Eu sou a Chica, sua assistente. Como posso ajudar?"

# Avatar (opcional — economiza memória)
AVATAR_ENABLE = True               # False para desabilitar

# Wake words
WAKE_WORDS = ["olá chica", "ei chica", "chica", ...]

# TTS (Kokoro / Edge / Qwen3)
TTS_SYSTEM = 'kokoro'             # 'kokoro', 'edge' ou 'qwen3'
TTS_VOICE = 'pf_dora'             # Kokoro: pf_dora (pt-BR)
EDGE_TTS_VOICE = 'pt-BR-ThalitaMultilingualNeural'  # Edge-TTS

# Sensibilidade do microfone (auto-detectado no macOS vs Linux/RPi)
SPEECH_THRESHOLD = 0.005
SILENCE_DURATION = 1.5
INACTIVITY_TIMEOUT = 15.0
AUDIO_DEVICE = "Isolamento de Voz"  # Auto: macOS → Isolamento de Voz, Linux → Padrão

# Speech-to-Text (Whisper / faster-whisper)
WHISPER_MODEL = 'turbo'            # 'tiny', 'base', 'small', 'medium', 'large', 'turbo'
WHISPER_LANGUAGE = 'pt'
STT_BACKEND = 'auto'               # 'auto' (detecta), 'whisper' (GPU), 'faster-whisper' (CPU)

# Memória
MEMORY_MAX_ITEMS = 30
MEMORY_CHAR_LIMIT = 2200
MEMORY_USER_CHAR_LIMIT = 1375
```

### Áudio: Detecção Automática de Plataforma

O dispositivo de áudio é selecionado automaticamente:

| Plataforma | `AUDIO_DEVICE` | Comportamento |
|---|---|---|
| **macOS** | `"Isolamento de Voz"` | Usa o filtro de ruído interno do macOS (Voice Isolation) |
| **Linux / Raspberry Pi** | `"Padrão"` | Usa o dispositivo padrão do sistema (ALSA/PulseAudio) |

Você pode sobrescrever manualmente em `config.py` com um nome específico (ex: `"USB Microphone"`).

### TTS: Fallback Automático

A Chica suporta 3 sistemas TTS:

| Sistema | Online? | Qualidade | Latência |
|---------|---------|-----------|----------|
| **Kokoro** (padrão) | ❌ Local | Boa | Baixa |
| **Edge-TTS** | ✅ Requer internet | Excelente (neural) | Média |
| **Qwen3-TTS** | ❌ Local | Avançada | Alta |

Se o **Edge-TTS** ou **Qwen3-TTS** falharem, o sistema faz **fallback automático** para Kokoro.

## 🚀 Executando

### 1. Inicie o provedor LLM:
- **LM Studio**: App aberto com API habilitada
- **Ollama**: `ollama serve`
- **llama.cpp**: `llama-server -m modelo.gguf -c 4096 --port 8080`

### 2. Execute a assistente:
```bash
python app.py
```

### 3. Na inicialização, escolha o TTS:
- `1` = Kokoro-TTS (padrão, português)
- `2` = Qwen3-TTS (voz Serena)
- `3` = Edge-TTS (voz Thalita, requer internet)
- `4` = Usar config.py

## 🗣️ Comandos de Voz

| Comando | Ação |
|---------|------|
| "olá chica" / "ei chica" | Acordar a assistente |
| "qual é a capital do Brasil?" | Conversa normal (via LLM) |
| "pesquise na internet sobre Python" | Pesquisa na web automática |
| "qual o tamanho do disco?" | Informações do sistema |
| "abra o navegador" | Abrir navegador (pede confirmação) |
| "abra o Spotify" | Abrir Spotify |
| "abra o VS Code" | Abrir Visual Studio Code |
| "abra o terminal" | Abrir terminal |
| "abra o Discord" | Abrir Discord |
| "calado" / "silêncio" | Interromper a fala |

### Detecção Inteligente de Intenção

A Chica usa um sistema de **2 camadas** para evitar falsos positivos em comandos locais:

1. **Heurísticas** (rápida, sem LLM): Detecta quando o usuário está perguntando sobre um app, não mandando abri-lo
   - *"calendário **dos** jogos"* → pergunta (não abre Calendar)
   - *"**qual** o melhor navegador?"* → pergunta
   - *"**pesquise** no navegador sobre python"* → pesquisa
2. **Verificação por LLM** (para casos ambíguos): Consulta o modelo local para classificar COMANDO vs PERGUNTA

Todos os comandos locais exigem confirmação por voz. Diga **"sim"** para confirmar, **"não"** para cancelar.

## 🧠 Memória Persistente v2

A Chica lembra de informações importantes entre conversas:

- **Extração automática** de fatos (nome, idade, preferências, gostos) em TODA interação via regex imediata
- **Extração via LLM** a cada interação para fatos complexos
- Armazenamento em **2 arquivos markdown** legíveis/editáveis:
  - `assistant_memory.md` — Fatos e memórias (limite: 2200 chars)
  - `assistant_user.md` — Perfil do usuário (limite: 1375 chars)
- Expira entradas antigas automaticamente quando estoura o limite
- Injetado no contexto do LLM — sem impacto na performance

## 📁 Estrutura do Projeto

```
kokoro/
├── app.py              (~1760 linhas)  — Entry point principal
├── config.py           (~ 322 linhas)  — Configurações centralizadas
├── llm_client.py       (~ 324 linhas)  — Cliente Ollama / LM Studio / llama.cpp
├── tts_engine.py       (~ 317 linhas)  — Síntese de voz (Kokoro / Edge / Qwen3)
├── audio_detector.py   (~ 344 linhas)  — Captura e detecção de voz (Whisper)
├── commands.py         (~ 240 linhas)  — Comandos locais + detecção de intenção
├── memory_manager.py   (~ 186 linhas)  — Memória persistente v2
├── system_info.py      (~ 250 linhas)  — Consultas de sistema (disco, RAM, CPU, IP)
├── avatar.py           (~ 252 linhas)  — Avatar Pygame animado
├── log.py              (~ 109 linhas)  — Logging colorido estruturado
├── chica_img/          ─ Imagens do avatar (PNG)
├── assistant_memory.md   ─ Memórias salvas (auto-gerado)
├── assistant_user.md     ─ Perfil do usuário (auto-gerado)
├── requirements.txt    ─ Dependências Python
├── leiame.txt          ─ Manual em português
└── README.md           ─ Este arquivo
```

### Módulos do Sistema

| Módulo | Função |
|--------|--------|
| `app.py` | Orquestrador principal: áudio → STT → LLM → TTS → avatar |
| `config.py` | Todas as configurações centralizadas |
|| `llm_client.py` | Interface com Ollama, LM Studio e llama.cpp |
| `tts_engine.py` | Síntese de voz (Kokoro, Edge-TTS, Qwen3) com fallback |
| `audio_detector.py` | Captura de áudio, detecção de fala, wake word, STT (Whisper) |
| `commands.py` | Execução de comandos locais com detecção inteligente de intenção |
| `memory_manager.py` | Memória persistente v2 em markdown |
| `system_info.py` | Consultas de informações do sistema |
| `avatar.py` | Animação do avatar em Pygame |
| `log.py` | Logging colorido com níveis |

## 🔄 Fluxo do Sistema

```
🎤 Microfone → Whisper (STT) → Detecção de Intenção
                               ├── Comando Local → Confirmação → Execução
                               ├── Pesquisa Web → LLM + Contexto
                               ├── Info Sistema → LLM + Contexto
                               └── Conversa → LLM
→ TTS (Kokoro/Edge/Qwen3) → 🔊 Alto-falante + 👤 Avatar
```

### Fluxo Detalhado:
1. **Wake word** detecta ativação
2. **Whisper** (`turbo`) transcreve áudio para texto
3. **Detecção de intenção** classifica:
   - Comando local? → Heurísticas + LLM verificam → pede confirmação → executa
   - Pesquisa web? → `search_web()` → injeta resultados no contexto
   - Info sistema? → `system_info.py` → injeta dados no contexto
   - Conversa geral? → Envia direto para o LLM
4. **LLM** (LM Studio / Ollama / llama.cpp) gera resposta
5. **TTS** converte resposta em áudio
6. **Avatar** anima sincronizado com o áudio

## 🎯 Dicas de Uso

1. Fale claramente em um ambiente silencioso
2. Aguarde a resposta em áudio antes de falar de novo
3. Diga **"calado"** para interromper a fala a qualquer momento
4. O avatar ajuda a visualizar quando a assistente está falando
5. Para economizar memória: `AVATAR_ENABLE = False` em `config.py`
6. O Whisper `turbo` oferece o melhor custo-benefício entre precisão e velocidade
7. **STT automático**: No Mac, usa whisper original com GPU (MPS). No SBC/ARM, usa faster-whisper com CTranslate2 (int8). Tudo automático via `STT_BACKEND='auto'`
8. Os modelos ficam em cache:
   - Whisper original: `~/.cache/whisper/` (~2.2 GB para o `turbo`)
   - faster-whisper: `~/.cache/huggingface/hub/` (mesmo modelo, formato CTranslate2)

## 🍓 Raspberry Pi / Orange Pi (SBC)

Recomendações para rodar em SBCs ARM:

### Instalação de áudio (uma vez)
```bash
sudo apt install pulseaudio pulseaudio-module-bluetooth libportaudio2 sox
pulseaudio --start
```

### Configuração
- O **faster-whisper** é selecionado automaticamente (MPS não disponível) — 4x mais rápido que whisper original em CPU
- O **dispositivo de áudio** é `"Padrão"` automaticamente (funciona com fone Bluetooth, USB, P2)
- Use Whisper `tiny` ou `base` se a RAM for limitada: `WHISPER_MODEL = 'base'`
- No **RPi 4 (4GB)**: recomendo `WHISPER_MODEL = 'base'` (turbo pesa muito)
- No **RPi 5 / Orange Pi 5**: `WHISPER_MODEL = 'turbo'` roda bem com CTranslate2

## 🛠️ Comandos Suportados (Sistema)

Tabela completa de aplicativos que podem ser abertos por voz:

| Fala | Ação (macOS) |
|------|-------------|
| "navegador" / "browser" | Safari |
| "chrome" | Google Chrome |
| "firefox" | Firefox |
| "safari" | Safari |
| "edge" | Microsoft Edge |
| "brave" | Brave Browser |
| "terminal" / "prompt" | Terminal |
| "vscode" / "vs code" | Visual Studio Code |
| "cursor" | Cursor |
| "spotify" | Spotify |
| "vlc" | VLC |
| "calculadora" | Calculator |
| "calendário" | Calendar |
| "notas" | Notes |
| "lembretes" | Reminders |
| "discord" | Discord |
| "telegram" | Telegram |
| "whatsapp" | WhatsApp |
| "notion" | Notion |
| "obsidian" | Obsidian |
| "finder" / "explorer" / "arquivos" | Finder |

## 🤝 Contribuindo

Para reportar problemas ou sugerir melhorias:
1. Verifique se o problema já foi reportado
2. Inclua informações do sistema
3. Descreva passos para reproduzir
4. Inclua mensagens de erro completas
