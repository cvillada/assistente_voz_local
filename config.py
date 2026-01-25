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

# ============================================================================
# CONFIGURAÇÕES DE VOZ TTS
# ============================================================================

# Configuração de voz TTS
# Formato: 'voz1' ou 'voz1 X% mais voz2 Y%' (ex: 'pf_dora 80% mais if_sara 20%')
TTS_VOICE = 'pf_dora'  # Vozes confirmadas: pf_dora (português), af_heart/af_bella/af_jessica (inglês)
TTS_SPEED = 1.0                      # Velocidade da fala (1.0 = normal, <1.0 = mais lento, >1.0 = mais rápido)

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
# CONFIGURAÇÕES DO MODELO OLLAMA
# ============================================================================

OLLAMA_MODEL = 'qwen3:1.7b'         # Modelo Ollama a ser usado "Não esqueça de ajustar o THINKING_ENABLED"
OLLAMA_TEMPERATURE = 0.7             # Criatividade (0.0 = determinístico, 1.0 = criativo)
OLLAMA_NUM_PREDICT = 300             # Número máximo de tokens na resposta

# ============================================================================
# CONFIGURAÇÃO DE THINKING (PROCESSAMENTO DO MODELO)
# ============================================================================

THINKING_ENABLED = True              # Ativa/desativa o modo thinking do modelo True ou False
THINKING_TIMEOUT = 30                # Timeout em segundos para processamento do thinking

# ============================================================================
# CONFIGURAÇÕES DO MODELO WHISPER
# ============================================================================

WHISPER_MODEL = 'base'                # Modelo Whisper: 'tiny', 'base', 'small', 'medium', 'large', 'turbo'

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