#!/usr/bin/env python3
"""
Configurações do Assistente de Voz Chica
Este módulo contém todas as configurações do sistema.
"""

# ============================================================================
# CONFIGURAÇÕES GERAIS DO SISTEMA
# ============================================================================

# Nome da assistente
ASSISTANT_NAME = "Chica"

# Saudação inicial da assistente
ASSISTANT_GREETING = f" Eu sou a {ASSISTANT_NAME}, sua assistente. Como posso ajudar?"

# Abre janela do avatar
AVATAR_ENABLE = True  #True a janela do Avatar é abilitada, False a janela do Avatar é desabilitada

# ============================================================================
# CONFIGURAÇÕES DO AVATAR
# ============================================================================

# Tamanho da janela do avatar (em centímetros)
AVATAR_WINDOW_SIZE_CM = 7  # 7cm x 7cm

# Conversão de cm para pixels (assumindo 96 DPI)
CM_TO_PIXELS = 37.8

# Tamanho mínimo da janela em pixels
AVATAR_MIN_SIZE_PX = 265

# Intervalo entre piscadas (segundos)
AVATAR_BLINK_INTERVAL = 3.0

# Duração da piscada (segundos)
AVATAR_BLINK_DURATION = 0.2

# Velocidade da animação de fala
AVATAR_SPEAK_ANIMATION_SPEED = 0.1

# Pasta das imagens do avatar
AVATAR_IMAGE_DIR = "chica_img"

# Estados das imagens do avatar
AVATAR_STATES = {
    "normal": "chica_normal.png",
    "olho": "chica_olho.png",
    "boca": "chica_boca.png",
    "olho_boca": "chica_olho_boca.png"
}

# ============================================================================
# CONFIGURAÇÕES DE ÁUDIO E SENSIBILIDADE
# ============================================================================

# Configurações básicas de áudio
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024

# Configurações de dispositivo de áudio (MacBook)
# ----------------------------------------------------------------------------
# No MacBook, você pode escolher entre:
# - "Padrão" (default): Usa o dispositivo padrão do sistema
# - "Isolamento de Voz" (voice_isolation): Filtra ruído de fundo
# - Especificar um dispositivo manualmente (ex: "MacBook Pro Microfone")
# ----------------------------------------------------------------------------
AUDIO_DEVICE = "Isolamento de Voz"  # Opções: "Padrão", "Isolamento de Voz", ou nome específico

# Configurações de sensibilidade de voz
# ----------------------------------------------------------------------------
# AJUSTE ESTES VALORES PARA OTIMIZAR A DETECÇÃO DE VOZ NO SEU AMBIENTE
# ----------------------------------------------------------------------------

# 1. DETECÇÃO DE FALA BÁSICA
INITIAL_NOISE_FLOOR = 0.001          # Piso de ruído inicial (quanto menor, mais sensível)
SPEECH_THRESHOLD = 0.005             # Limiar fixo para detecção de fala (quanto menor, mais sensível)
MIN_SPEECH_DURATION = 0.3            # Duração mínima para considerar como fala (segundos)
SILENCE_DURATION = 1.5               # Segundos de silêncio para disparar processamento

# 2. PARÂMETROS ADAPTATIVOS
NOISE_FLOOR_UPDATE_THRESHOLD = 1.5   # Multiplicador para atualizar piso de ruído (1.5 = atualiza quando RMS < 1.5x threshold)
NOISE_FLOOR_SMOOTHING = 0.8          # Suavização do piso de ruído (0.8 = 80% valor anterior + 20% novo)
DYNAMIC_THRESHOLD_MULTIPLIER = 2.0   # Multiplicador para limiar dinâmico (noise_floor * X)
SPEECH_ENERGY_MULTIPLIER = 1.2       # Multiplicador para verificar fala legítima (threshold * X)

# 3. CONFIGURAÇÕES DE TEMPO
INACTIVITY_TIMEOUT = 15.0            # Segundos sem atividade para dormir

# 4. CONFIGURAÇÕES DE INTERAÇÃO
WAKE_WORDS = [
    "olá chica",     # Original
    "ei chica",      # Informal
    "chica",         # Apenas o nome
    "ok chica",      # Com "ok"
    "shika",         # Pronúncia alternativa
    "shica",         # Pronúncia alternativa 2
    "hei chica",     # Pronúncia em inglês
    "hei shica",     # Pronúncia alternativa em inglês
]

# Palavras que interrompem a fala da assistente
STOP_PHRASES = [
    "calado",
    "calada",
    "silêncio",
    "silencio",
    "para",
    "pare",
    "chega",
    "basta",
    "stop",
]

# ============================================================================
# CONFIGURAÇÕES DE VOZ TTS
# ============================================================================

# Configuração de voz TTS
# Formato: 'voz1' ou 'voz1 X% mais voz2 Y%' (ex: 'pf_dora 80% mais if_sara 20%')
TTS_VOICE = 'pf_dora'  # Vozes confirmadas: pf_dora (português), af_heart/af_bella/af_jessica (inglês)
TTS_SPEED = 1.05                      # Velocidade da fala (1.0 = normal, <1.0 = mais lento, >1.0 = mais rápido)

# Taxa de amostragem do TTS (Hz)
TTS_SAMPLE_RATE = 24000              # Kokoro gera áudio a 24kHz

# Modelo Kokoro (HuggingFace repo_id)
# Opções comuns: 'hexgrad/Kokoro-82M' (padrão, 82M params),
#                'hexgrad/Kokoro-82M-v2.0' (versão mais nova)
TTS_KOKORO_MODEL = 'hexgrad/Kokoro-82M'

# Código de idioma do Kokoro
# 'p' = português, 'a' = inglês americano, 'b' = inglês britânico,
# 'j' = japonês, 'k' = coreano, 'z' = chinês, 'f' = francês, 'e' = espanhol
TTS_KOKORO_LANG = 'p'

# ============================================================================
# CONFIGURAÇÕES DO SISTEMA TTS (KOKORO vs QWEN3)
# ============================================================================

# Sistema TTS a ser usado
# Opções: 'kokoro' (padrão) ou 'qwen3'
TTS_SYSTEM = 'kokoro'

# Configurações específicas do Qwen3-TTS
QWEN3_MODEL = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'  # Modelo Qwen3-TTS
QWEN3_VOICE = 'serena'  # Voz para português (serena)
QWEN3_LANGUAGE = 'portuguese'  # Idioma para síntese (em inglês: 'portuguese')
QWEN3_USE_COMPILE = False  # torch.compile piora desempenho no MPS (testes mostraram 10x mais lento)

# ============================================================================
# CONFIGURAÇÕES DO EDGE-TTS (ONLINE — REQUER INTERNET)
# ============================================================================

# Sistema TTS: 'kokoro' (padrão, local), 'qwen3' (local) ou 'edge' (online)
# Edge-TTS usa a API gratuita do Microsoft Edge para síntese neural.
# Requer internet. Se ficar offline, faz fallback automático para Kokoro.
#TTS_SYSTEM = 'edge'

# Voz padrão do Edge-TTS (pt-BR)
# Lista completa de vozes disponíveis:
#   pt-BR-ThalitaMultilingualNeural  (Feminino, natural) ← recomendada
#   pt-BR-AntonioNeural              (Masculino)
#   pt-BR-FranciscaNeural            (Feminino)
#
# Outras vozes em inglês (caso mude o idioma):
#   en-US-AriaNeural       (Feminino)
#   en-US-GuyNeural        (Masculino)
#   en-GB-SoniaNeural      (Feminino, britânico)
#   en-GB-RyanNeural       (Masculino, britânico)
EDGE_TTS_VOICE = 'pt-BR-ThalitaMultilingualNeural'

# Velocidade da fala: 0.5 (lento) a 2.0 (rápido), padrão 1.0
EDGE_TTS_SPEED = 1.0

# ============================================================================
# CONFIGURAÇÕES DO PROVEDOR LLM
# ============================================================================

# Provedor de linguagem: 'ollama', 'lm_studio' ou 'llamacpp'
#   ollama   → via servidor Ollama local (porta 11434)
#   lm_studio → via LM Studio local (porta 1234)
#   llamacpp  → via llama-server local (porta 8080, servidor OpenAI-compatible)
LLM_PROVIDER = 'lm_studio'

# Modelo a ser usado (para Ollama: 'qwen3:1.7b', 'llama3.2:3b', etc.
# Para LM Studio: nome do modelo carregado na interface, ex: 'llama-3.2-3b-instruct'
# Para llama.cpp: nome usado no servidor, ex: 'qwen2.5-3b-instruct')
LLM_MODEL = 'bonsai-8b'

# ============================================================================
# CONFIGURAÇÕES DO LM STUDIO (usado apenas quando LLM_PROVIDER = 'lm_studio')
# ============================================================================

LM_STUDIO_HOST = 'localhost'         # Host do servidor LM Studio
LM_STUDIO_PORT = 1234                # Porta do servidor LM Studio (padrão: 1234)

# ============================================================================
# CONFIGURAÇÕES DO LLAMA.CPP (usado apenas quando LLM_PROVIDER = 'llamacpp')
# ============================================================================

LLAMACPP_HOST = 'localhost'          # Host do llama-server (ex: 'localhost' ou IP do RPi)
LLAMACPP_PORT = 8080                 # Porta do llama-server (padrão: 8080)

# ============================================================================
# CONFIGURAÇÕES GERAIS DO MODELO (aplicam-se a ambos os provedores)
# ============================================================================

LLM_TEMPERATURE = 0.7                # Criatividade (0.0 = determinístico, 1.0 = criativo)
LLM_NUM_PREDICT = 300                # Número máximo de tokens na resposta

# Aliases para compatibilidade com versões anteriores
OLLAMA_MODEL = LLM_MODEL
OLLAMA_TEMPERATURE = LLM_TEMPERATURE
OLLAMA_NUM_PREDICT = LLM_NUM_PREDICT

# ============================================================================
# CONFIGURAÇÃO DE THINKING (PROCESSAMENTO DO MODELO - apenas Ollama)
# ============================================================================

THINKING_ENABLED = True              # Ativa/desativa o modo thinking do modelo True ou False
THINKING_TIMEOUT = 30                # Timeout em segundos para processamento do thinking

# ============================================================================
# CONFIGURAÇÕES DE MEMÓRIA PERSISTENTE
# ============================================================================

# Número máximo de itens em cada arquivo de memória
# (tanto assistant_memory.md quanto assistant_user.md)
# Quando estourar, remove os mais antigos automaticamente
MEMORY_MAX_ITEMS = 30

# Limite de caracteres para cada arquivo de memória (estilo Hermes)
# Quando estourar, remove entradas mais antigas automaticamente
MEMORY_CHAR_LIMIT = 2200       # assistant_memory.md (~800 tokens)
MEMORY_USER_CHAR_LIMIT = 1375  # assistant_user.md (~500 tokens)

# ============================================================================
# CONFIGURAÇÕES DO MODELO WHISPER
# ============================================================================

WHISPER_MODEL = 'turbo'                # Modelo: 'tiny', 'base', 'small', 'medium', 'large', 'turbo'
WHISPER_LANGUAGE = 'pt'               # Idioma: 'pt' (português), 'en' (inglês), 'es' (espanhol), etc.

# Backend do STT: 'auto' (detecta hardware), 'whisper' (original, PyTorch), 'faster-whisper' (CTranslate2)
# - Mac Apple Silicon → 'whisper' (usa MPS/GPU, mais rápido)
# - ARM / SBC / CPU → 'faster-whisper' (CTranslate2 otimizado)
# - NVIDIA GPU → 'faster-whisper' (suporta CUDA)
STT_BACKEND = 'auto'

# ============================================================================
# FUNÇÕES AUXILIARES DE CONFIGURAÇÃO
# ============================================================================

def get_avatar_window_size():
    """
    Calcula o tamanho da janela do avatar em pixels.
    Retorna uma tupla (width, height).
    """
    # Converter cm para pixels
    width = int(AVATAR_WINDOW_SIZE_CM * CM_TO_PIXELS)
    height = int(AVATAR_WINDOW_SIZE_CM * CM_TO_PIXELS)
    
    # Garantir tamanhos mínimos
    width = max(width, AVATAR_MIN_SIZE_PX)
    height = max(height, AVATAR_MIN_SIZE_PX)
    
    return (width, height)

def parse_voice_config(voice_config):
    """
    Processa a configuração de voz para mesclagem.
    
    Formato aceito:
    - 'voz1' (ex: 'pf_dora')
    - 'voz1 X% mais voz2 Y%' (ex: 'pf_dora 80% mais if_sara 20%')
    
    Retorna:
    - Para voz única: {'voz1': 100}
    - Para mesclagem: {'voz1': percentual1, 'voz2': percentual2}
    """
    import re
    voice_config = voice_config.strip().lower()
    
    # Verificar se é uma configuração de mesclagem
    if 'mais' in voice_config and '%' in voice_config:
        try:
            # Extrair as partes
            parts = voice_config.split('mais')
            if len(parts) != 2:
                return {'pf_dora': 100}  # Fallback
            
            # Processar primeira voz
            part1 = parts[0].strip()
            voice1_match = re.search(r'([a-z_]+)\s+(\d+)%', part1)
            if not voice1_match:
                return {'pf_dora': 100}  # Fallback
            
            voice1 = voice1_match.group(1)
            percent1 = int(voice1_match.group(2))
            
            # Processar segunda voz
            part2 = parts[1].strip()
            voice2_match = re.search(r'([a-z_]+)\s+(\d+)%', part2)
            if not voice2_match:
                return {'pf_dora': 100}  # Fallback
            
            voice2 = voice2_match.group(1)
            percent2 = int(voice2_match.group(2))
            
            # Verificar se a soma é 100%
            if percent1 + percent2 != 100:
                return {'pf_dora': 100}  # Fallback
            
            return {voice1: percent1, voice2: percent2}
            
        except Exception:
            return {'pf_dora': 100}  # Fallback em caso de erro
    
    # Se for apenas uma voz
    else:
        return {voice_config: 100}