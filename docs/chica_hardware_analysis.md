# Análise de Hardware — Chica Voice Assistant em SBC

## Setup Atual

- **LLM**: bonsai-4b (quantização 1-bit) → ~570 MB em disco / ~700 MB em RAM
- **STT**: large-v3-turbo. No Mac usa whisper original com GPU (MPS). No SBC usa **faster-whisper** (CTranslate2, int8)
- **Kokoro-82M**: ~160 MB em disco / ~300 MB em RAM
- **Edge-TTS**: 0 recursos locais (só internet)

## Consumo de RAM (SBC ARM — faster-whisper ativado)

O STT agora **detecta automaticamente** o hardware. No SBC (sem GPU), ele usa faster-whisper com CTranslate2, que consome ~1/3 da RAM do whisper original:

| Componente | RAM (faster-whisper int8) |
|---|---|
| SO Linux ARM64 leve (Ubuntu Server, sem desktop) | ~300 MB |
| SO + LXDE (desktop leve + telinha) | ~500 MB |
| bonsai-4b 1-bit (llama.cpp) | ~700 MB |
| faster-whisper large-v3-turbo (int8) | **~500 MB** ⬅️ |
| Kokoro-82M | ~400 MB |
| Avatar Pygame + SDL2 | ~200 MB |
| **TOTAL (com avatar, sem desktop pesado)** | **~2.100 MB** |
| **TOTAL (com desktop leve + avatar)** | **~2.500 MB** |

> ✅ Com **4GB de RAM**, sobra ~1.5 GB para o sistema — **confortável**.
> ✅ Com **8GB**, sobra ~5.5 GB — **folgado**.

A economia de ~1GB veio da troca do whisper original (PyTorch, ~1.5GB) pelo faster-whisper (CTranslate2 int8, ~500MB).

## STT no Mac vs SBC (automático — sem config)

A Chica já tem `STT_BACKEND = 'auto'` em config.py. A detecção funciona sozinha:

| Hardware | Backend escolhido | RAM | Velocidade |
|---|---|---|---|
| Mac Apple Silicon | whisper original (MPS/GPU) | ~1.5 GB | Rápido (GPU) |
| Raspberry Pi 5 (ARM) | faster-whisper (CPU int8) | ~500 MB | 4x mais rápido que whisper original em CPU |
| Orange Pi 5 (ARM) | faster-whisper (CPU int8) | ~500 MB | 4x mais rápido |
| Orange Pi 5 (GPU Vulkan) | faster-whisper (CPU int8) | ~500 MB | Pode usar GPU se configurado |

## Comparativo de Hardware

### CPU Performance (llama.cpp, bonsai-4b 1-bit)

| Dispositivo | tokens/s (CPU) | tokens/s (GPU/NPU) |
|---|---|---|
| **Raspberry Pi 4** (4x A72 @ 1.8GHz) | 3-5 tok/s | — |
| **Raspberry Pi 5** (4x A76 @ 2.4GHz) | **6-10 tok/s** | — |
| **Orange Pi 5** (4x A76 + 4x A55) | 8-12 tok/s | **20-30 tok/s** (GPU Vulkan) |
| **Orange Pi 5** (com NPU Rockchip) | — | **30-40 tok/s** (NPU) |

### STT Performance (large-v3-turbo, transcrição de 5s de áudio)

| Dispositivo | Whisper original | faster-whisper int8 |
|---|---|---|
| **Mac Apple Silicon** | ~0.5s (MPS/GPU) 🚀 | — (não usa no Mac) |
| **Raspberry Pi 5** | ~4.0s (CPU) | **~1.0s** 🚀 |
| **Orange Pi 5** | ~3.0s (CPU) | **~0.8s** 🚀 |

### Matriz de Compatibilidade

| Componente | RPi 4 (4GB) | RPi 5 (8GB) | Orange Pi 5 (8GB) | Orange Pi 5 (16GB) |
|---|---|---|---|---|
| **bonsai-4b 1-bit** | ✅ 3-5 tok/s | ✅ 6-10 tok/s | ✅ 8-12 CPU / 20+ GPU | ✅ folgado |
| **STT (auto → faster-whisper)** | ✅ ~1.5s áudio | ✅ ~1.0s áudio | ✅ ~0.8s áudio | ✅ folgado |
| **Kokoro TTS** | ✅ leve | ✅ leve | ✅ leve | ✅ leve |
| **Edge-TTS** | ✅ (só internet) | ✅ (só internet) | ✅ (só internet) | ✅ (só internet) |
| **Avatar + telinha** | ✅ (CPU) | ✅ (CPU) | ✅ (GPU Mali) | ✅ (GPU Mali) |
| **Tudo simultâneo** | ✅ (4GB, sobra ~1.5GB) | ✅✅ (8GB) | ✅✅ (8GB) | ✅✅✅ (16GB) |

> ⚠️ **Nota sobre RPi 4 (4GB)**: Antes da troca para faster-whisper, o RPi 4 ficava no limite (~3.3GB de ~4GB). Agora com ~2.5GB, sobra 1.5GB — **rodou tranquilo**.

## Telinhas Compatíveis

### Raspberry Pi 5
- **HDMI**: Tela HDMI 3.5" a 7" (micro-HDMI para HDMI)
- **DSI**: Display oficial Raspberry Pi 7" (conector DSI, ribbon cable)
- **GPIO**: Telas SPI (ILI9341, 3.5") — lentas, mas OK para avatar simples

### Orange Pi 5
- **HDMI**: Telas HDMI (fácil, qualquer tamanho)
- **MIPI DSI nativo**: Conector de 40 pinos — telas de 5" a 10.1" disponíveis
- **GPIO**: Telas SPI

> **Vantagem Orange Pi 5**: MIPI DSI nativo (mais rápido que DSI do Pi e mais fácil de achar telas).

## Modelos de LLM Alternativos

| Modelo | Tamanho | RAM | tok/s (Pi 5) | tok/s (Orange 5) |
|---|---|---|---|---|
| **bonsai-4b 1-bit** | 570 MB | ~700 MB | 6-10 | 8-12 CPU / 20+ GPU |
| **Qwen2.5-1.5B Q4_K_M** | ~1 GB | ~1.2 GB | 8-12 | 12-18 CPU |
| **Llama-3.2-3B Q4_K_M** | ~2 GB | ~2.2 GB | 4-6 | 6-10 CPU |
| **Qwen2.5-0.5B Q4_K_M** | ~350 MB | ~450 MB | 15-25 | 20-30 CPU |

## Recomendação Final

### 🥇 Orange Pi 5 (8GB) — ~R$ 750

**Melhor custo-benefício.**

- Roda tudo com folga (sobra ~5.5GB de RAM)
- GPU Mali pode acelerar LLM via Vulkan (20+ tok/s)
- MIPI DSI nativo para telinha (sem adaptador)
- STT automático usa faster-whisper (~0.8s por transcrição)
- 4x A76 (performance) + 4x A55 (economia)

### 🥈 Raspberry Pi 5 (8GB) — ~R$ 650

**Mais fácil de configurar, maior comunidade.**

- Roda tudo (sobra ~5.5GB)
- LLM mais lento (6-10 tok/s), mas **usável** para respostas curtas
- STT automático usa faster-whisper (~1s por transcrição)
- Precisa de tela HDMI ou adaptador DSI
- Setup mais simples (Raspberry Pi OS)

### 🥉 Raspberry Pi 5 (16GB) — ~R$ 950

- Mesma CPU do Pi 5 8GB
- Mais RAM (se quiser modelos maiores futuramente)

### ⛔ Raspberry Pi 4 (4GB) — Agora é viável com faster-whisper!

Antes estava no limite (~3.3GB). Agora com faster-whisper (~2.5GB total), sobra ~1.5GB:
- ✅ LLM mais lento (3-5 tok/s), mas funcional
- ✅ STT com faster-whisper (~1.5s por transcrição)
- ✅ Kokoro (leve)
- ⚠️ Avatar pode ser pesado — considere desligar se travar

## Custo Estimado

```
=== ORÇAMENTO (Orange Pi 5 8GB) ===
Orange Pi 5 (8GB)                R$ 750
Fonte USB-C 5V/4A                R$  60
Cartão microSD 128GB (A2)        R$  80
Tela MIPI DSI 5" 800x480        R$ 120
Case acrílico + dissipador       R$  50
Microfone USB                   R$  40
Caixinha de som USB              R$  30
-------------------------------------
TOTAL                           R$ 1.130

=== ORÇAMENTO (Raspberry Pi 5 8GB) ===
Raspberry Pi 5 (8GB)             R$ 650
Fonte USB-C 27W                  R$  70
Cartão microSD 128GB (A2)        R$  80
Case oficial + cooler            R$  80
Tela HDMI 5" 800x480             R$ 110
Microfone USB                    R$  40
Caixinha de som USB              R$  30
-------------------------------------
TOTAL                           R$ 1.060
```

## Config para o SBC

```python
# config.py — SBC (funciona sem alterações, STT_BACKEND='auto' já detecta)
LLM_PROVIDER = 'custom'           # llama.cpp server
LLM_MODEL = 'bonsai-4b-Q2_K.gguf' # ~570MB
WHISPER_MODEL = 'turbo'           # faster-whisper gerencia em int8
STT_BACKEND = 'auto'              # detecta hardware → escolhe faster-whisper
TTS_SYSTEM = 'kokoro'            # local, leve
AVATAR_ENABLE = True
THINKING_ENABLED = False          # ganha velocidade
```

## Áudio: Entrada e Saída

### Microfone (Entrada)

Para o STT (Whisper/faster-whisper) funcionar bem, o microfone é **crítico**. Opções:

| Tipo | Preço | Qualidade | Prós | Contras |
|---|---|---|---|---|
| **Microfone USB genérico** | R$ 30-50 | Razoável | Plug-and-play, já incluso no orçamento | Pega ruído ambiente |
| **Webcam com microfone** | R$ 80-150 | Boa | Já serve pra video se quiser | Mais caro |
| **Headset USB** | R$ 60-100 | Boa | Isolamento acústico, não ecoa | Desconfortável para uso prolongado |
| **Microfone condensador USB** (Fifine, BM-800) | R$ 80-120 | Excelente | Qualidade profissional | Ocupa espaço |
| **Microfone embutido do case** (se comprar um case com microfone) | — | Variável | Integrado | Depende do case |

> **Recomendação para protótipo**: Microfone USB genérico de R$ 40 já funciona. Se quiser melhorar depois, um Fifine USB de R$ 80 faz diferença.

### Alto-falante (Saída)

Para o TTS (Kokoro/Edge-TTS), a saída de áudio também importa:

| Tipo | Preço | Prós | Contras |
|---|---|---|---|
| **Caixinha de som USB** | R$ 25-40 | Já no orçamento, plug-and-play | Qualidade de som básica |
| **HDMI audio** (pela tela) | R$ 0 (já tem) | Zero custo extra | Depende da tela ter speaker |
| **P2/PWM + mini speaker** | R$ 15-30 | Pode usar GPIO, mais barato | Precisa de conversor, mais trabalho |
| **Bluetooth speaker** | R$ 60-150 | Sem fio, qualidade melhor | Latência maior (pode dessincronizar com o avatar) |

> **Recomendação para protótipo**: Caixinha USB de R$ 30. Simples, funciona, sem configurar nada.

### Onde conectar no SBC?

**Orange Pi 5 / Raspberry Pi 5:**

```
USB ── Microfone USB (entrada)
USB ── Caixinha de som USB (saída)
HDMI ── Tela (vídeo + áudio, se a tela tiver speaker)
GPIO ── PWM + mini speaker (avançado)
```

Ambos têm:
- **USB 3.0** — suficiente para microfone + caixinha simultaneamente
- **P2 (PCM)** — saída de áudio analógica de 3.5mm (RPi 5 tem, Orange Pi 5 tem via GPIO)
- **HDMI** — áudio digital embutido no cabo HDMI

### Configuração de áudio no Linux

No SBC, o `sounddevice` (usado pela Chica) lista os dispositivos automaticamente. Para ver:

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

No `config.py` da Chica, você pode especificar:

```python
# Para usar o microfone USB (descubra o nome com o comando acima)
AUDIO_DEVICE = "USB Microphone"  # ou "default" para o padrão
```

### Cuidado: Loop de Áudio (Eco)

Quando a Chica fala pelo alto-falante e o microfone capta o som dela mesma, pode criar eco. Soluções:
1. **Não use o mesmo dispositivo para entrada e saída** (microfone USB + caixinha USB separados → OK)
2. O **VAD filter** do faster-whisper (`vad_filter=True`) ajuda a ignorar eco
3. No Orange Pi 5, dá para usar o **P2 output** (fone de ouvido) + microfone USB para isolar

## Pipeline de Instalação

```bash
# Orange Pi 5 / Raspberry Pi 5
sudo apt update
sudo apt install python3-pip portaudio19-dev libsdl2-dev
pip install faster-whisper kokoro edge-tts openai

# Compilar llama.cpp (Orange Pi 5: com Vulkan para GPU)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DLLAMA_VULKAN=ON       # Orange Pi 5
make -j4

# Baixar bonsai-4b GGUF e iniciar servidor
./llama-server -m bonsai-4b-Q2_K.gguf -c 4096 --port 8080

# Rodar Chica
cd /home/user/kokoro/
python3 app.py
```

## Nota: STT Automático

A Chica já tem `STT_BACKEND = 'auto'` configurado. No SBC, ele detecta que não há MPS disponível e usa **faster-whisper** automaticamente — **não precisa configurar nada**. O mesmo código funciona no Mac (usa whisper original com GPU) e no SBC (usa faster-whisper com CPU).
