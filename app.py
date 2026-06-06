import whisper
import torch
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import tempfile
import time
import sounddevice as sd
import queue
import threading
import os
import wave
from collections import deque
from colorama import init, Fore, Style
import signal
import sys
import re
import warnings

# Importações para o avatar
import pygame
from pygame.locals import *

# Gerenciador do avatar animado
from avatar import AvatarManager

# Motor TTS (Kokoro / Qwen3)
from tts_engine import TTSManager

# Detector de áudio e voz
from audio_detector import AudioDetector

# Executor de comandos locais
from commands import CommandExecutor

# Memória persistente
from memory_manager import MemoryManager

# Logging colorido
from log import logger

# Cliente LLM abstrato (Ollama / LM Studio)
from llm_client import LLMClient, LLMError

# Importar configurações do módulo config
import config

# Filtrar warnings de forma abrangente para reduzir poluição no terminal
warnings.filterwarnings("ignore", category=Warning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Filtrar warnings específicos da biblioteca kokoro
warnings.filterwarnings("ignore", 
                       message=".*dropout option adds dropout after all but last recurrent layer.*")
warnings.filterwarnings("ignore", 
                       message=".*torch.nn.utils.weight_norm is deprecated.*")
warnings.filterwarnings("ignore",
                       message=".*Defaulting repo_id to hexgrad/Kokoro-82M.*")
warnings.filterwarnings("ignore", 
                       category=FutureWarning,
                       module="torch.nn.utils.weight_norm")

# Filtrar warnings específicos do Qwen3-TTS e Transformers
warnings.filterwarnings("ignore", 
                       message=".*Setting `pad_token_id` to `eos_token_id`.*")
warnings.filterwarnings("ignore",
                       message=".*`max_length` is set.*")

# Inicializar colorama para cores no terminal
init(autoreset=True)

# ============================================================================
# CONFIGURAÇÕES IMPORTADAS DO MÓDULO CONFIG
# ============================================================================

# Configurações gerais
ASSISTANT_NAME = config.ASSISTANT_NAME
ASSISTANT_GREETING = config.ASSISTANT_GREETING

# Configurações de áudio
SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = config.CHANNELS
CHUNK = config.CHUNK
AUDIO_DEVICE = config.AUDIO_DEVICE

# Configurações de sensibilidade de voz
INITIAL_NOISE_FLOOR = config.INITIAL_NOISE_FLOOR
SPEECH_THRESHOLD = config.SPEECH_THRESHOLD
MIN_SPEECH_DURATION = config.MIN_SPEECH_DURATION
SILENCE_DURATION = config.SILENCE_DURATION
NOISE_FLOOR_UPDATE_THRESHOLD = config.NOISE_FLOOR_UPDATE_THRESHOLD
NOISE_FLOOR_SMOOTHING = config.NOISE_FLOOR_SMOOTHING
DYNAMIC_THRESHOLD_MULTIPLIER = config.DYNAMIC_THRESHOLD_MULTIPLIER
SPEECH_ENERGY_MULTIPLIER = config.SPEECH_ENERGY_MULTIPLIER
INACTIVITY_TIMEOUT = config.INACTIVITY_TIMEOUT
WAKE_WORDS = config.WAKE_WORDS

# Configurações de voz TTS
TTS_VOICE = config.TTS_VOICE
TTS_SPEED = config.TTS_SPEED
TTS_SAMPLE_RATE = config.TTS_SAMPLE_RATE

# Configurações do sistema TTS
TTS_SYSTEM = config.TTS_SYSTEM
QWEN3_MODEL = config.QWEN3_MODEL
QWEN3_VOICE = config.QWEN3_VOICE
QWEN3_LANGUAGE = config.QWEN3_LANGUAGE

# Configurações dos modelos
OLLAMA_MODEL = config.OLLAMA_MODEL          # alias, prefira LLM_MODEL
OLLAMA_TEMPERATURE = config.OLLAMA_TEMPERATURE
OLLAMA_NUM_PREDICT = config.OLLAMA_NUM_PREDICT
WHISPER_MODEL = config.WHISPER_MODEL

# Configurações do provedor LLM
LLM_PROVIDER = config.LLM_PROVIDER
LLM_MODEL = config.LLM_MODEL
LLM_TEMPERATURE = config.LLM_TEMPERATURE
LLM_NUM_PREDICT = config.LLM_NUM_PREDICT
LM_STUDIO_HOST = config.LM_STUDIO_HOST
LM_STUDIO_PORT = config.LM_STUDIO_PORT

# Configurações de thinking
THINKING_ENABLED = config.THINKING_ENABLED
THINKING_TIMEOUT = config.THINKING_TIMEOUT

# ============================================================================
# CLASSE PARA GERENCIAR O AVATAR
# ============================================================================

# ============================================================================
# FUNÇÕES AUXILIARES PARA PROCESSAMENTO DE VOZ
# ============================================================================

# parse_voice_config está definida em config.py — importar de lá
from config import parse_voice_config

class ChicaAssistant:
    def __init__(self):
        print(Fore.CYAN + "="*60)
        print(Fore.YELLOW + f"ASSISTENTE {ASSISTANT_NAME} - MODO POR VOZ")
        print(Fore.CYAN + "="*60)
        
        # Opção de seleção de sistema TTS
        print(Fore.CYAN + "🎤 SELECIONE O SISTEMA TTS:")
        print(Fore.CYAN + "   1. Kokoro-TTS (padrão)")
        print(Fore.CYAN + "   2. Qwen3-TTS (voz Serena em português)")
        print(Fore.CYAN + "   3. Usar configuração atual do config.py")
        
        try:
            choice = input(Fore.YELLOW + "Escolha (1/2/3) [3]: ").strip()
            if choice == "1":
                self.tts_system_choice = 'kokoro'
                print(Fore.GREEN + "✅ Sistema TTS selecionado: Kokoro-TTS")
            elif choice == "2":
                self.tts_system_choice = 'qwen3'
                print(Fore.GREEN + "✅ Sistema TTS selecionado: Qwen3-TTS (voz Serena)")
            else:
                self.tts_system_choice = TTS_SYSTEM
                print(Fore.GREEN + f"✅ Usando configuração do config.py: {TTS_SYSTEM}")
        except (KeyboardInterrupt, EOFError):
            print(Fore.YELLOW + "\n⚠️  Usando configuração padrão do config.py")
            self.tts_system_choice = TTS_SYSTEM
        
        print(Fore.CYAN + "-"*60)
        
        # Mostrar configurações de voz
        print(Fore.MAGENTA + f"🔊 Configuração de voz:")
        
        # Mostrar sistema TTS selecionado
        print(Fore.MAGENTA + f"   • Sistema TTS: {self.tts_system_choice}")
        
        if self.tts_system_choice == 'kokoro':
            # Processar e mostrar configuração de voz kokoro
            voice_config = parse_voice_config(TTS_VOICE)
            if len(voice_config) == 1:
                voice_name = list(voice_config.keys())[0]
                percent = list(voice_config.values())[0]
                print(Fore.MAGENTA + f"   • Voz: {voice_name} ({percent}%)")
            else:
                voices = list(voice_config.keys())
                percents = list(voice_config.values())
                voice_desc = f"{voices[0]} {percents[0]}% + {voices[1]} {percents[1]}%"
                print(Fore.MAGENTA + f"   • Vozes mescladas: {voice_desc}")
            
            print(Fore.MAGENTA + f"   • Velocidade: {TTS_SPEED}")
            print(Fore.MAGENTA + f"   • Vozes confirmadas: pf_dora (pt), af_heart/af_bella/af_jessica (en)")
            print(Fore.MAGENTA + f"   • Para alterar, edite TTS_VOICE e TTS_SPEED nas linhas 73-74")
            print(Fore.MAGENTA + f"   • Formato: 'voz1' ou 'voz1 X% mais voz2 Y%'")
        else:
            # Mostrar configurações do Qwen3-TTS
            print(Fore.MAGENTA + f"   • Modelo: {QWEN3_MODEL}")
            print(Fore.MAGENTA + f"   • Voz: {QWEN3_VOICE}")
            print(Fore.MAGENTA + f"   • Idioma: {QWEN3_LANGUAGE}")
            print(Fore.MAGENTA + f"   • Para alterar, edite TTS_SYSTEM, QWEN3_MODEL, QWEN3_VOICE e QWEN3_LANGUAGE nas linhas 79-84")
        
        print(Fore.CYAN + "-"*60)
        
        # Mostrar configurações de sensibilidade
        print(Fore.YELLOW + f"🎯 Configurações de sensibilidade:")
        print(Fore.YELLOW + f"   • Piso de ruído: {INITIAL_NOISE_FLOOR}")
        print(Fore.YELLOW + f"   • Limiar de fala: {SPEECH_THRESHOLD}")
        print(Fore.YELLOW + f"   • Duração mínima: {MIN_SPEECH_DURATION}s")
        print(Fore.YELLOW + f"   • Silêncio para processar: {SILENCE_DURATION}s")
        print(Fore.YELLOW + f"   • Para ajustar, edite as linhas 48-62")
        print(Fore.CYAN + "-"*60)
        
        # Mostrar configurações de áudio
        print(Fore.BLUE + f"🔊 Configurações de áudio:")
        print(Fore.BLUE + f"   • Dispositivo: {AUDIO_DEVICE}")
        print(Fore.BLUE + f"   • Taxa de amostragem: {SAMPLE_RATE} Hz")
        print(Fore.BLUE + f"   • Canais: {CHANNELS}")
        print(Fore.BLUE + f"   • Tamanho do chunk: {CHUNK}")
        print(Fore.BLUE + f"   • Para alterar, edite AUDIO_DEVICE na linha 49")
        print(Fore.CYAN + "-"*60)
        
        # Mostrar configurações do provedor LLM
        provider_label = "Ollama" if LLM_PROVIDER == 'ollama' else "LM Studio"
        print(Fore.CYAN + f"🤖 Configurações do provedor LLM:")
        print(Fore.CYAN + f"   • Provedor: {provider_label}")
        print(Fore.CYAN + f"   • Modelo: {LLM_MODEL}")
        print(Fore.CYAN + f"   • Temperatura: {LLM_TEMPERATURE}")
        print(Fore.CYAN + f"   • Máximo de tokens: {LLM_NUM_PREDICT}")
        if LLM_PROVIDER == 'lm_studio':
            print(Fore.CYAN + f"   • Host: {LM_STUDIO_HOST}:{LM_STUDIO_PORT}")
        print(Fore.CYAN + f"   • Thinking: {'✅ ativado' if THINKING_ENABLED else '❌ desativado'} (apenas Ollama)")
        print(Fore.CYAN + f"   • Para alterar, edite LLM_PROVIDER e LLM_MODEL no config.py")
        print(Fore.CYAN + "-"*60)
        
        # Mostrar configurações do modelo Whisper
        print(Fore.MAGENTA + f"🎤 Configurações do Whisper:")
        print(Fore.MAGENTA + f"   • Modelo: {WHISPER_MODEL}")
        print(Fore.MAGENTA + f"   • Opções: 'tiny', 'base', 'small', 'medium', 'large', 'turbo'")
        print(Fore.MAGENTA + f"   • Para alterar, edite WHISPER_MODEL na linha 95")
        print(Fore.CYAN + "-"*60)
        
        # Carregar modelos
        print(Fore.GREEN + "Inicializando modelos...")
        
        # Verificar dispositivo de processamento
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Dispositivo de processamento: {device}")
        
        # Configurar dispositivo de áudio
        self.audio_device_id = self._get_audio_device_id()
        
        # Carregar modelos
        # Modelos maiores que 'base' têm problemas conhecidos com MPS
        models_with_mps_issues = ['small', 'medium', 'large', 'turbo']
        
        if WHISPER_MODEL in models_with_mps_issues and device == "mps":
            print(Fore.YELLOW + f"⚠️  Modelo '{WHISPER_MODEL}' tem problemas conhecidos com MPS")
            print(Fore.YELLOW + f"⚠️  Carregando em CPU para evitar erros de precisão...")
            device = "cpu"
        
        try:
            self.stt_model = whisper.load_model(WHISPER_MODEL, device=device)
            print(Fore.GREEN + f"✅ Modelo Whisper '{WHISPER_MODEL}' carregado com sucesso em {device}")
        except Exception as e:
            # Se falhar, tentar com CPU
            print(Fore.YELLOW + f"⚠️  Erro ao carregar modelo Whisper em {device}: {str(e)[:100]}...")
            print(Fore.YELLOW + f"⚠️  Tentando carregar em CPU...")
            try:
                self.stt_model = whisper.load_model(WHISPER_MODEL, device="cpu")
                print(Fore.GREEN + f"✅ Modelo Whisper '{WHISPER_MODEL}' carregado com sucesso em CPU")
            except Exception as e2:
                print(Fore.RED + f"❌ Erro crítico ao carregar modelo Whisper: {str(e2)[:100]}...")
                print(Fore.RED + f"❌ Tentando carregar modelo 'tiny' como fallback...")
                self.stt_model = whisper.load_model("tiny", device="cpu")
                print(Fore.GREEN + f"✅ Modelo Whisper 'tiny' carregado como fallback em CPU")
        
        # Inicializar sistema TTS baseado na escolha do usuário
        self.tts_system = self.tts_system_choice
        self.tts_pipeline = None
        self.qwen3_pipeline = None
        
        if self.tts_system == 'kokoro':
            print(Fore.GREEN + "🔊 Inicializando sistema TTS Kokoro...")
            self.tts_pipeline = KPipeline(lang_code='p', repo_id='hexgrad/Kokoro-82M')
            print(Fore.GREEN + "✅ Sistema TTS Kokoro inicializado com sucesso")
        else:
            self._init_qwen3_tts()

        # Inicializar cliente LLM (Ollama ou LM Studio)
        print(Fore.CYAN + "🤖 Inicializando cliente LLM...")
        try:
            self.llm = LLMClient(
                provider=LLM_PROVIDER,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_NUM_PREDICT,
                lm_studio_host=LM_STUDIO_HOST,
                lm_studio_port=LM_STUDIO_PORT,
            )
            provider_display = "Ollama" if LLM_PROVIDER == 'ollama' else f"LM Studio ({LM_STUDIO_HOST}:{LM_STUDIO_PORT})"
            print(Fore.GREEN + f"✅ Cliente LLM inicializado: {provider_display} › {LLM_MODEL}")
        except LLMError as e:
            print(Fore.YELLOW + f"⚠️  Aviso ao inicializar LLM: {e}")
            print(Fore.YELLOW + "⚠️  O assistente pode não funcionar corretamente sem um provedor LLM.")

        # Estado
        self.conversation_history = []
        self.audio_buffer = []
        self.is_listening = True
        self.is_processing = False
        self.is_speaking_tts = False  # Novo: indica se a IA está falando
        self.last_speech_time = time.time()
        self.last_activity_time = time.time()
        self.is_active = False  # Começa inativa, precisa de wake word
        self.wake_word_detected = False
        self.inactivity_counter = INACTIVITY_TIMEOUT  # Contador de inatividade
        
        # Avatar
        if config.AVATAR_ENABLE:
            self.avatar = AvatarManager()
            self.avatar_started = False
            
            # Iniciar avatar imediatamente
            self.start_avatar()
        else:
            self.avatar = None
            self.avatar_started = False
            print(Fore.YELLOW + "⚠️  Avatar desabilitado (AVATAR_ENABLE = False)")
        
        # Parâmetros adaptativos de detecção
        self.noise_floor = INITIAL_NOISE_FLOOR
        self.speech_threshold = SPEECH_THRESHOLD
        self.user_is_speaking = False
        self.consecutive_speech_chunks = 0
        self.silence_chunks_needed = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
        self.silence_chunks_counter = 0
        
        # Dispositivo de áudio
        self.audio_device_id = None
        
        # Buffer para interrupções
        self.interruption_buffer = []
        self.interruption_enabled = True  # Permite interromper a IA
        self._checking_interruption = False  # Lock para evitar concorrência na interrupção

        # Feature: comandos locais
        self.command_executor = CommandExecutor()
        self.waiting_confirmation = None  # None ou dict do comando pendente

        # Memória persistente
        mem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chica_memory.json')
        self.memory = MemoryManager(mem_path)
        self._conversation_count = 0  # Para rate-limited memory extraction
        self.stop_phrases = ["calado", "calada", "silêncio", "silencio"]  # Apenas comandos explícitos de interrupção
        
        # Cache para frases curtas frequentes (melhora performance do Qwen3-TTS)
        self.tts_cache = {}
        self.max_cache_size = 50  # Limite máximo de frases em cache
        self.qwen3_warmed_up = False  # Flag para controle de pré-aquecimento
        
        # Configurar handler para CTRL+C
        signal.signal(signal.SIGINT, self.signal_handler)

    def _init_qwen3_tts(self) -> None:
        """Inicializa o TTS Qwen3 com fallback automático para Kokoro."""
        print(Fore.GREEN + f"🔊 Inicializando sistema TTS Qwen3 ({QWEN3_MODEL})...")
        try:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
            tts_device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Dispositivo TTS: {tts_device}")
            if tts_device == "mps":
                self.qwen3_pipeline = Qwen3TTSModel.from_pretrained(QWEN3_MODEL, device_map="mps")
            else:
                self.qwen3_pipeline = Qwen3TTSModel.from_pretrained(QWEN3_MODEL)
            actual_device = (
                self.qwen3_pipeline.model.device
                if hasattr(self.qwen3_pipeline.model, 'device')
                else next(self.qwen3_pipeline.model.parameters()).device
            )
            print(Fore.GREEN + f"✅ Sistema TTS Qwen3 inicializado com sucesso (dispositivo: {actual_device})")
            print(Fore.GREEN + f"   • Voz: {QWEN3_VOICE}")
            print(Fore.GREEN + f"   • Idioma: {QWEN3_LANGUAGE}")

            if config.QWEN3_USE_COMPILE and hasattr(torch, 'compile'):
                try:
                    print(Fore.YELLOW + "   • Compilando modelo com torch.compile...")
                    compiled_model = torch.compile(self.qwen3_pipeline.model, mode="reduce-overhead")
                    self.qwen3_pipeline.model = compiled_model
                    print(Fore.GREEN + "   • Modelo compilado com sucesso")
                except Exception as compile_e:
                    print(Fore.YELLOW + f"   • Aviso na compilação: {compile_e}")

            try:
                print(Fore.YELLOW + "   • Pré-aquentendo modelo Qwen3-TTS...")
                _ = self.qwen3_pipeline.generate_custom_voice(
                    text="Olá", speaker=QWEN3_VOICE,
                    language=QWEN3_LANGUAGE, non_streaming_mode=True
                )
                self.qwen3_warmed_up = True
                print(Fore.GREEN + "   • Modelo pré-aquecido com sucesso")
            except Exception as warmup_e:
                print(Fore.YELLOW + f"   • Aviso no pré-aquecimento: {warmup_e}")
        except ImportError:
            print(Fore.RED + "❌ Erro: Qwen3-TTS não está instalado")
            print(Fore.YELLOW + "⚠️  Instale com: pip install qwen-tts")
            self._fallback_to_kokoro()
        except Exception as e:
            print(Fore.YELLOW + f"⚠️  Aviso ao inicializar Qwen3-TTS: {e}")
            print(Fore.YELLOW + "⚠️  Tentando novamente...")
            try:
                from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
                tts_device = "mps" if torch.backends.mps.is_available() else "cpu"
                if tts_device == "mps":
                    self.qwen3_pipeline = Qwen3TTSModel.from_pretrained(QWEN3_MODEL, device_map="mps")
                else:
                    self.qwen3_pipeline = Qwen3TTSModel.from_pretrained(QWEN3_MODEL)
                if config.QWEN3_USE_COMPILE and hasattr(torch, 'compile'):
                    try:
                        compiled_model = torch.compile(self.qwen3_pipeline.model, mode="reduce-overhead")
                        self.qwen3_pipeline.model = compiled_model
                    except Exception:
                        pass
                try:
                    _ = self.qwen3_pipeline.generate_custom_voice(
                        text="Olá", speaker=QWEN3_VOICE,
                        language=QWEN3_LANGUAGE, non_streaming_mode=True
                    )
                    self.qwen3_warmed_up = True
                except Exception:
                    pass
                print(Fore.GREEN + "✅ Qwen3-TTS inicializado na segunda tentativa")
            except Exception:
                print(Fore.RED + "❌ Não foi possível inicializar Qwen3-TTS")
                self._fallback_to_kokoro()

    def _fallback_to_kokoro(self) -> None:
        """Fallback para Kokoro-TTS quando Qwen3 falha."""
        print(Fore.YELLOW + "⚠️  Usando Kokoro-TTS como fallback...")
        self.tts_system = 'kokoro'
        self.tts_pipeline = KPipeline(lang_code='p', repo_id='hexgrad/Kokoro-82M')
        print(Fore.GREEN + "✅ Sistema TTS Kokoro inicializado como fallback")

    def _get_audio_device_id(self):
        """
        Encontra o ID do dispositivo de áudio baseado na configuração AUDIO_DEVICE.
        Retorna None para usar o dispositivo padrão.
        """
        try:
            # Listar todos os dispositivos disponíveis
            devices = sd.query_devices()
            
            # Se for "Padrão", retorna None (usará o padrão do sistema)
            if AUDIO_DEVICE.lower() == "padrão" or AUDIO_DEVICE.lower() == "default":
                print(Fore.GREEN + "🎤 Usando dispositivo de áudio padrão do sistema")
                return None
            
            # Se for "Isolamento de Voz", procurar por dispositivos com esse nome
            if AUDIO_DEVICE.lower() == "isolamento de voz" or AUDIO_DEVICE.lower() == "voice isolation":
                # Procurar por dispositivos que contenham "isolamento", "voice isolation", etc.
                for i, device in enumerate(devices):
                    device_name = device['name'].lower()
                    if ('isolamento' in device_name or 
                        'voice isolation' in device_name or 
                        'voice_isolation' in device_name or
                        'voiceisolation' in device_name):
                        print(Fore.GREEN + f"✅ Encontrado dispositivo de isolamento de voz: {device['name']}")
                        return i
                
                # Se não encontrou isolamento de voz, usar o padrão
                print(Fore.YELLOW + "⚠️  Dispositivo 'Isolamento de Voz' não encontrado.")
                print(Fore.YELLOW + "🎤 Usando dispositivo de áudio padrão do sistema")
                return None
            
            # Procurar por nome exato ou parcial
            for i, device in enumerate(devices):
                if AUDIO_DEVICE.lower() in device['name'].lower():
                    print(Fore.GREEN + f"✅ Encontrado dispositivo: {device['name']}")
                    return i
            
            # Se não encontrou o dispositivo especificado, mostrar opções e usar padrão
            print(Fore.YELLOW + f"⚠️  Dispositivo '{AUDIO_DEVICE}' não encontrado.")
            print(Fore.YELLOW + "📋 Dispositivos de áudio disponíveis:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Apenas dispositivos de entrada
                    default_marker = " (padrão)" if i == sd.default.device[0] else ""
                    print(Fore.YELLOW + f"   [{i}] {device['name']}{default_marker}")
            
            print(Fore.YELLOW + "🎤 Usando dispositivo de áudio padrão do sistema")
            return None
            
        except Exception as e:
            print(Fore.RED + f"⚠️  Erro ao listar dispositivos de áudio: {e}")
            print(Fore.YELLOW + "🎤 Usando dispositivo de áudio padrão do sistema")
            return None
        
        print(Fore.GREEN + f"✓ {ASSISTANT_NAME} inicializada!")
        print(Fore.CYAN + "-"*40)
        print(Fore.YELLOW + f"\n🔇 {ASSISTANT_NAME} está dormindo...")
        print(Fore.CYAN + f"Diga '{WAKE_WORDS[0]}' para acordá-la (ou outras variações)")
        print(Fore.CYAN + f"Variações aceitas: {', '.join(WAKE_WORDS)}")
        print(Fore.CYAN + f"Após {INACTIVITY_TIMEOUT}s de silêncio, ela volta a dormir")
        print(Fore.CYAN + f"• Diga 'pare' ou 'para' para interromper a {ASSISTANT_NAME}")
        print(Fore.GREEN + f"• Sensibilidade: {SPEECH_THRESHOLD} (ajuste nas linhas 48-62 se necessário)")
        print(Fore.CYAN + "-"*40)
    
    def extract_ai_response(self, response):
        """
        Extrai a resposta da IA do objeto response do Ollama.
        Considera o campo 'thinking' se THINKING_ENABLED for True.
        """
        try:
            # Acessar o objeto message da resposta
            message = response.message
            
            # Verificar se temos conteúdo no campo content
            if hasattr(message, 'content') and message.content and message.content.strip():
                return message.content.strip()
            
            # Se não tem conteúdo, verificar se temos thinking
            if THINKING_ENABLED and hasattr(message, 'thinking') and message.thinking:
                # O thinking contém o processamento interno do modelo
                # Precisamos extrair a resposta final do thinking
                thinking_text = message.thinking
                
                # Estratégia 1: Procurar por "Final answer:" ou similar
                # Buscar a ÚLTIMA ocorrência para pegar apenas a resposta final
                final_answer_patterns = [
                    'final answer:', 'resposta final:', 'answer:', 'resposta:',
                    'responder:', 'conclusão:', 'portanto', 'assim', 'dessa forma'
                ]
                
                # Converter para minúsculas para busca case-insensitive
                thinking_lower = thinking_text.lower()
                
                # Encontrar a posição da última ocorrência de qualquer padrão
                last_pattern_pos = -1
                last_pattern = ""
                
                for pattern in final_answer_patterns:
                    pos = thinking_lower.rfind(pattern)
                    if pos > last_pattern_pos:
                        last_pattern_pos = pos
                        last_pattern = pattern
                
                # Se encontramos um padrão, extrair o texto após ele
                if last_pattern_pos != -1:
                    # Encontrar a posição real no texto original
                    # Precisamos encontrar a ocorrência correspondente no texto original
                    pattern_pos = thinking_text.lower().rfind(last_pattern)
                    if pattern_pos != -1:
                        # Extrair texto após o padrão
                        start_pos = pattern_pos + len(last_pattern)
                        extracted = thinking_text[start_pos:].strip()
                        
                        # Limpar: remover pontuação inicial e espaços extras
                        extracted = extracted.lstrip(' :.-')
                        
                        # Verificar se a resposta extraída não é uma repetição óbvia
                        # de uma resposta anterior (como no cenário problemático)
                        if extracted:
                            # Verificar se a resposta parece ser uma repetição
                            # de uma resposta de saudação ou identidade
                            repetition_indicators = [
                                'eu sou a chica', 'sou azul porque', 
                                'minha identidade', 'como posso ajudar',
                                'sua assistente'
                            ]
                            
                            # Se a resposta extraída contém indicadores de repetição
                            # E o thinking menciona "previous answer" ou similar,
                            # devemos tentar uma estratégia diferente
                            extracted_lower = extracted.lower()
                            is_possible_repetition = any(
                                indicator in extracted_lower 
                                for indicator in repetition_indicators
                            )
                            
                            # Verificar se o thinking menciona resposta anterior
                            has_previous_ref = any(
                                phrase in thinking_lower 
                                for phrase in ['previous answer', 'last answer', 'already answered', 'similar question']
                            )
                            
                            # Se parece ser uma repetição E há referência a resposta anterior,
                            # usar estratégia alternativa
                            if is_possible_repetition and has_previous_ref:
                                print(Fore.YELLOW + "⚠️  Possível repetição detectada, usando estratégia alternativa...")
                                # Continuar para as próximas estratégias
                            else:
                                return extracted
                
                # Estratégia 2: Dividir por linhas e procurar a última linha significativa
                lines = thinking_text.split('\n')
                cleaned_lines = []
                
                # Padrões que indicam raciocínio interno (para remover)
                reasoning_patterns = [
                    'okay,', 'first,', 'next,', 'then,', 'now,',
                    'i need to', 'i should', 'let me', 'i think',
                    'the user', 'user asked', 'user said',
                    'wait,', 'but', 'actually,', 'let me check',
                    'what was', 'last answer', 'previous answer',
                    'already answered', 'similar question'
                ]
                
                # Padrões que indicam resposta (para priorizar)
                answer_patterns = [
                    'final answer', 'resposta', 'answer', 'conclusão',
                    'portanto', 'assim', 'dessa forma', 'logo'
                ]
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Remover linhas que são claramente raciocínio interno
                    line_lower = line.lower()
                    if any(pattern in line_lower for pattern in reasoning_patterns):
                        continue
                    
                    # Verificar se a linha parece ser uma resposta
                    is_answer_line = any(pattern in line_lower for pattern in answer_patterns)
                    
                    # Se for uma linha de resposta, marcar com prioridade
                    if is_answer_line:
                        # Extrair apenas a parte da resposta (após o padrão)
                        for pattern in answer_patterns:
                            if pattern in line_lower:
                                pattern_pos = line_lower.find(pattern)
                                if pattern_pos != -1:
                                    # Extrair texto após o padrão
                                    answer_part = line[pattern_pos + len(pattern):].strip()
                                    answer_part = answer_part.lstrip(' :.-')
                                    if answer_part:
                                        cleaned_lines.append(('answer', answer_part))
                                        break
                        else:
                            cleaned_lines.append(('normal', line))
                    else:
                        cleaned_lines.append(('normal', line))
                
                # Se temos linhas limpas, analisar para encontrar a melhor resposta
                if cleaned_lines:
                    # Primeiro, procurar por linhas marcadas como 'answer'
                    answer_lines = [text for type_, text in cleaned_lines if type_ == 'answer']
                    if answer_lines:
                        # Pegar a última linha de resposta
                        return answer_lines[-1].strip()
                    
                    # Se não encontrou linhas de resposta explícitas,
                    # analisar o conteúdo das últimas linhas
                    
                    # Pegar as últimas 2-3 linhas para análise
                    last_items = cleaned_lines[-3:] if len(cleaned_lines) >= 3 else cleaned_lines
                    last_texts = [text for _, text in last_items]
                    
                    # Juntar as últimas linhas
                    candidate = ' '.join(last_texts).strip()
                    
                    # Verificar se o candidato parece ser uma resposta válida
                    # (não muito curto, não parece ser raciocínio)
                    if len(candidate) > 10 and not any(
                        pattern in candidate.lower() 
                        for pattern in ['the user', 'user asked', 'i need to', 'let me']
                    ):
                        return candidate
                    
                    # Se o candidato não parece bom, tentar todas as linhas limpas
                    all_texts = [text for _, text in cleaned_lines]
                    fallback = ' '.join(all_texts).strip()
                    if fallback:
                        return fallback
                
                # Estratégia 3: Fallback - usar o thinking completo mas limpo
                cleaned_thinking = thinking_text
                
                # Remover prefixos comuns de thinking
                prefixes_to_remove = [
                    'Okay, ', 'First, ', 'Next, ', 'Then, ', 'Now, ',
                    'I need to ', 'I should ', 'Let me ', 'I think ',
                    'The user ', 'User asked', 'User said'
                ]
                
                for prefix in prefixes_to_remove:
                    if cleaned_thinking.startswith(prefix):
                        cleaned_thinking = cleaned_thinking[len(prefix):]
                
                return cleaned_thinking.strip()
            
            # Se não temos nem content nem thinking, retornar mensagem padrão
            return "Desculpe, não consegui processar a resposta."
            
        except Exception as e:
            print(Fore.YELLOW + f"⚠️  Erro ao extrair resposta: {e}")
            # Fallback: tentar o método antigo
            try:
                return response['message']['content'].strip()
            except:
                return "Desculpe, tive um problema ao processar a resposta."

    def clean_text_for_tts(self, text):
        """Limpa o texto removendo emojis e caracteres especiais indesejados"""
        if not text:
            return ""
        
        # Remover emojis e caracteres especiais Unicode
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        
        # Remover caracteres especiais
        caracteres_para_remover = [
            '*', '~', '_', '`', '|', '•', '·', '©', '®', '™',
            '→', '←', '↑', '↓', '↔', '↕', '⇒', '⇐', '⇑', '⇓',
            '∞', '≠', '≤', '≥', '≈', '≡', '≅', '∀', '∃', '∄',
            '∅', '∆', '∇', '∈', '∉', '∋', '∌', '∏', '∑', '√',
            '∛', '∜', '∝', '∞', '∟', '∠', '∧', '∨', '∩', '∪',
            '∫', '∬', '∭', '∮', '∴', '∵', '∶', '∷', '∼', '∽'
        ]
        
        for char in caracteres_para_remover:
            text = text.replace(char, '')
        
        # Remover múltiplos espaços em branco
        text = re.sub(r'\s+', ' ', text)
        
        # Remover parênteses e conteúdo dentro
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r'\{[^}]*\}', '', text)
        
        # Remover aspas especiais
        text = text.replace('"', '').replace("'", '')
        
        # Substituir caracteres problemáticos
        text = text.replace('*', '').replace('_', ' ').replace('~', '').replace('`', '')
        
        # Corrigir múltiplas exclamações ou interrogações
        text = re.sub(r'\!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        # Remover traços especiais
        text = text.replace('—', ', ').replace('–', ', ').replace('−', '-')
        
        # Remover marcadores de lista
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'^[\s]*[•\-*\d\.\)]+[\s]*', '', line)
            if line.strip():
                cleaned_lines.append(line.strip())
        
        text = ' '.join(cleaned_lines)
        
        # Garantir que termine com pontuação
        text = text.strip()
        if text and not text[-1] in '.!?':
            text = text + '.'
        
        # Limpar espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_text_for_display(self, text):
        """Limpa o texto para exibição no terminal (menos agressivo)"""
        if not text:
            return ""
        
        # Remover apenas os piores emojis para display
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        
        # Substituir alguns caracteres por versões mais amigáveis
        text = text.replace('*', '').replace('~', '').replace('_', ' ')
        
        return text
    
    def check_for_stop_command(self, text):
        """Verifica se o texto contém um comando para parar"""
        text_lower = text.lower().strip()
        
        for phrase in self.stop_phrases:
            if phrase in text_lower:
                return True
        
        # Verificar padrões como "Chica, para" ou "Chica pare"
        words = text_lower.split()
        if len(words) >= 2:
            if ASSISTANT_NAME.lower() in words[0] and words[1] in ["calado", "calada", "silêncio"]:
                return True
        
        return False

    def _ask_command_confirmation(self, cmd: dict) -> None:
        """Pergunta confirmação para executar um comando local."""
        confirm_text = cmd["confirmacao"]
        print(Fore.CYAN + f"\n🤖 {ASSISTANT_NAME}: {confirm_text}")
        self.waiting_confirmation = cmd
        clean_for_tts = self.clean_text_for_tts(confirm_text)
        audio_file = self.text_to_speech(clean_for_tts)
        if audio_file:
            self.play_audio_with_interruption(audio_file)

    def _handle_confirmation_response(self, text: str) -> None:
        """Processa a resposta do usuário à confirmação."""
        text_lower = text.lower().strip()
        cmd = self.waiting_confirmation
        self.waiting_confirmation = None

        confirm_words = ["sim", "confirmo", "pode", "pode abrir", "ok", "tá", "claro", "vai", "pode ir"]
        reject_words = ["não", "nao", "cancela", "cancelar", "pare", "para", "nada", "nenhum"]

        if any(w in text_lower for w in confirm_words):
            result = self.command_executor.execute(cmd)
            print(Fore.GREEN + f"\n✅ {result}")
            clean = self.clean_text_for_tts(result)
            audio_file = self.text_to_speech(clean)
            if audio_file:
                self.play_audio_with_interruption(audio_file)
        elif any(w in text_lower for w in reject_words):
            msg = "Comando cancelado."
            print(Fore.YELLOW + f"\n🤖 {ASSISTANT_NAME}: {msg}")
            audio_file = self.text_to_speech(self.clean_text_for_tts(msg))
            if audio_file:
                self.play_audio_with_interruption(audio_file)
        else:
            msg = "Não entendi. Diga 'sim' para confirmar ou 'não' para cancelar."
            print(Fore.YELLOW + f"\n🤖 {ASSISTANT_NAME}: {msg}")
            self.waiting_confirmation = cmd  # Re-ask
            audio_file = self.text_to_speech(self.clean_text_for_tts(msg))
            if audio_file:
                self.play_audio_with_interruption(audio_file)

    def _extract_memory(self, user_text: str, ai_reply: str) -> None:
        """Extrai fatos importantes da conversa e salva na memória.

        Executa a cada 5 interações para não impactar performance.
        Usa uma chamada LLM leve e curta.
        """
        try:
            prompt = (
                'Extraia informações importantes sobre o usuário desta conversa. '
                'Retorne uma frase curta (máx 15 palavras) ou "NONE" se nada relevante. '
                'Armazene apenas fatos úteis para conversas futuras '
                '(nome, preferências, informações pessoais relevantes).'
            )
            resp = self.llm.chat([
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': user_text},
                {'role': 'assistant', 'content': ai_reply},
            ])
            fact = resp.message.content.strip()
            if fact and fact.upper() != 'NONE':
                self.memory.add(fact)
        except Exception as e:
            logger.warning(f"Erro na extração de memória: {e}")

    def signal_handler(self, sig, frame):
        """Handler para CTRL+C"""
        print(Fore.RED + "\n\n🛑 Interrompendo...")
        self.is_listening = False
        sys.exit(0)
    
    def check_wake_word(self, text):
        """Verifica se o texto contém a wake word"""
        text_lower = text.lower().strip()
        
        # Remover pontuação comum para melhor análise
        import string
        text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
        words = text_clean.split()
        
        # Palavras que precisam de contexto (precedidas por prefixo ou serem a primeira palavra)
        words_need_context = ["shika", "shica", "chica"]
        
        # Verificar cada wake word da lista
        for wake_word in WAKE_WORDS:
            # Se a wake word está no texto (com ou sem pontuação)
            if wake_word in text_lower or wake_word in text_clean:
                # Se for uma palavra que precisa de contexto
                if any(word in words_need_context for word in wake_word.split()):
                    # Verificar se está no início ou precedida por prefixo
                    wake_word_parts = wake_word.split()
                    for i in range(len(words) - len(wake_word_parts) + 1):
                        if words[i:i+len(wake_word_parts)] == wake_word_parts:
                            # Verificar se está no início ou precedida por prefixo comum
                            if i == 0:
                                return True
                            # Prefixos comuns
                            prefixes = ["olá", "ola", "oi", "ei", "hey", "hei", "ok", "okay", "tá", "ta", "pronto"]
                            if words[i-1] in prefixes:
                                return True
                else:
                    # Para outras wake words, aceitar em qualquer posição
                    return True
        
        # Verificar variações com acentos removidos
        for wake_word in WAKE_WORDS:
            wake_word_no_accents = wake_word.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
            if wake_word_no_accents in text_lower or wake_word_no_accents in text_clean:
                # Mesma lógica de contexto para palavras que precisam
                if any(word in words_need_context for word in wake_word_no_accents.split()):
                    wake_word_parts = wake_word_no_accents.split()
                    for i in range(len(words) - len(wake_word_parts) + 1):
                        if words[i:i+len(wake_word_parts)] == wake_word_parts:
                            if i == 0:
                                return True
                            prefixes = ["olá", "ola", "oi", "ei", "hey", "hei", "ok", "okay", "tá", "ta", "pronto"]
                            if words[i-1] in prefixes:
                                return True
                else:
                    return True
        
        return False
    
    def check_inactivity(self):
        """Verifica inatividade e coloca para dormir se necessário"""
        if self.is_processing or self.is_speaking_tts:
            return False
            
        current_time = time.time()
        inactivity_duration = current_time - self.last_activity_time
        
        if self.is_active and inactivity_duration > INACTIVITY_TIMEOUT:
            print(Fore.YELLOW + f"\n💤 {ASSISTANT_NAME} está dormindo...")
            print(Fore.CYAN + f"Diga '{WAKE_WORDS[0]}' para acordá-la (ou outras variações)")
            self.is_active = False
            self.conversation_history = []
            return True
        return False
    
    def reset_inactivity_counter(self):
        """Reseta o contador de inatividade para o valor inicial"""
        self.inactivity_counter = INACTIVITY_TIMEOUT
        self.last_activity_time = time.time()
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback para processamento de áudio em tempo real"""
        if status:
            return
        
        # Se a IA está falando, não processar áudio normal
        if self.is_speaking_tts:
            # Mas ainda escutamos para interrupções
            if self.interruption_enabled:
                self.interruption_buffer.append(indata.copy())
                
                # Limitar o buffer de interrupção
                max_interruption_duration = 3.0
                max_interruption_size = int(max_interruption_duration * SAMPLE_RATE / CHUNK)
                if len(self.interruption_buffer) > max_interruption_size:
                    self.interruption_buffer = self.interruption_buffer[-max_interruption_size:]
            return
        
        # Adicionar ao buffer normal
        audio_chunk = indata.copy()
        
        # Calcular energia RMS do chunk
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Atualizar piso de ruído adaptativamente
        if rms < self.speech_threshold * NOISE_FLOOR_UPDATE_THRESHOLD:
            # Atualização do piso de ruído durante silêncio
            self.noise_floor = self.noise_floor * NOISE_FLOOR_SMOOTHING + rms * (1 - NOISE_FLOOR_SMOOTHING)
        
        # Ajustar limiar de fala dinamicamente baseado no piso de ruído
        dynamic_threshold = max(self.speech_threshold, self.noise_floor * DYNAMIC_THRESHOLD_MULTIPLIER)
        
        # Verificar se é fala do usuário
        is_speech_now = rms > dynamic_threshold
        
        if is_speech_now:
            # Está falando agora
            self.consecutive_speech_chunks += 1
            self.silence_chunks_counter = 0
            
            if not self.user_is_speaking and self.consecutive_speech_chunks > 1:  # Reduzido de 2 para 1
                # Começou a falar mais rapidamente
                self.user_is_speaking = True
                self.last_speech_time = time.time()
                if self.is_active:
                    self.last_activity_time = time.time()
            
            # Adicionar ao buffer se estiver falando
            self.audio_buffer.append(audio_chunk)
            
            # Limitar tamanho do buffer para evitar consumo excessivo de memória
            max_buffer_duration = 10.0  # 10 segundos máximo
            max_buffer_size = int(max_buffer_duration * SAMPLE_RATE / CHUNK)
            if len(self.audio_buffer) > max_buffer_size:
                self.audio_buffer = self.audio_buffer[-max_buffer_size:]
            
        else:
            # Silêncio agora
            self.consecutive_speech_chunks = 0
            self.silence_chunks_counter += 1
            
            if self.user_is_speaking:
                # Ainda está no período de fala, continua adicionando ao buffer
                self.audio_buffer.append(audio_chunk)
                
                # Verificar se terminou de falar (silêncio suficiente)
                if self.silence_chunks_counter >= self.silence_chunks_needed:
                    # Terminou de falar, processar buffer
                    self.user_is_speaking = False
                    
                    # Verificar se há áudio suficiente para processar
                    buffer_duration = len(self.audio_buffer) * CHUNK / SAMPLE_RATE
                    if buffer_duration >= MIN_SPEECH_DURATION and not self.is_processing:
                        threading.Thread(target=self.process_audio_buffer, daemon=True).start()
    
    def process_audio_buffer(self):
        """Processa o buffer de áudio acumulado"""
        if not self.audio_buffer or self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            # Combinar buffers
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            
            # Verificar energia média do áudio
            audio_energy = np.sqrt(np.mean(audio_data**2))
            
            # Verificar se é fala legítima
            if audio_energy < self.speech_threshold * SPEECH_ENERGY_MULTIPLIER:
                # Muito fraco, provavelmente ruído
                self.audio_buffer.clear()
                self.is_processing = False
                return
            
            # Limpar buffer
            self.audio_buffer.clear()
            
            # Salvar em arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, audio_data, SAMPLE_RATE)
            
            # Processar áudio
            self.process_interaction(temp_file.name)
            
            # Limpar arquivo
            try:
                os.unlink(temp_file.name)
            except:
                pass
                
        except Exception as e:
            print(Fore.RED + f"Erro ao processar áudio: {e}")
        finally:
            self.is_processing = False
    
    def check_interruption(self):
        """Verifica se há interrupção enquanto a IA está falando.

        Usa janela deslizante dos últimos ~1s de áudio.
        NÃO limpa o buffer — o áudio que chega durante a transcrição
        (1-2s) é preservado e processado na próxima verificação.
        """
        if not self.interruption_buffer or self._checking_interruption:
            return False

        self._checking_interruption = True
        try:
            # Janela deslizante: apenas os últimos ~1s de áudio
            chunk_count = max(1, int(1.0 * SAMPLE_RATE / CHUNK))
            recent = self.interruption_buffer[-chunk_count:]

            # Só transcrever se houver áudio suficiente (mín 0.5s)
            min_chunks = int(0.5 * SAMPLE_RATE / CHUNK)
            if len(recent) < min_chunks:
                return False

            interruption_data = np.concatenate(recent, axis=0)

            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, interruption_data, SAMPLE_RATE)

            result = self.stt_model.transcribe(temp_file.name, language="pt")
            user_text = result["text"].strip().lower()

            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

            # NÃO limpar o buffer! O áudio que chegou durante a
            # transcrição será processado na próxima verificação.
            # A janela deslizante (last ~1s) evita re-processar
            # áudio velho.

            if user_text and self.check_for_stop_command(user_text):
                print(Fore.YELLOW + f"\n⏸️  {ASSISTANT_NAME} interrompida!")
                return True

        except Exception:
            pass
        finally:
            self._checking_interruption = False

        return False
    
    def process_interaction(self, audio_path):
        """Processa uma interação completa"""
        start_time = time.time()
        
        # 1. Transcrever áudio
        result = self.stt_model.transcribe(audio_path, language="pt")
        user_text = result["text"].strip()
        
        if not user_text:
            return
        
        print(Fore.BLUE + f"\n🎤 Você: {user_text}")
        
        # 2. Verificar wake word se estiver inativa
        if not self.is_active:
            if self.check_wake_word(user_text):
                print(Fore.GREEN + f"\n🔔 {ASSISTANT_NAME} acordou!")
                self.is_active = True
                self.reset_inactivity_counter()
                self.wake_word_detected = True
                
                # Saudação inicial (usando configuração do config.py)
                greeting = config.ASSISTANT_GREETING
                print(Fore.GREEN + f"🤖 {ASSISTANT_NAME}: {greeting}")
                
                # Converter para áudio
                clean_greeting = self.clean_text_for_tts(greeting)
                audio_file = self.text_to_speech(clean_greeting)
                if audio_file:
                    self.play_audio_with_interruption(audio_file)
                
                return
            else:
                return
        
        # 3. Verificar se é comando de parar
        if self.check_for_stop_command(user_text):
            print(Fore.YELLOW + f"\n⏸️  Comando de parar detectado")
            return

        # 3.5 Verificar se estamos aguardando confirmação de comando
        if self.waiting_confirmation is not None:
            self._handle_confirmation_response(user_text)
            return

        # 3.6 Detectar comandos locais (abrir navegador, etc.)
        cmd = self.command_executor.parse(user_text)
        if cmd:
            self._ask_command_confirmation(cmd)
            return

        # 4. Se estiver ativa, processar normalmente
        # Incluir memórias persistentes no system prompt
        memory_context = self.memory.get_context()
        system_prompt = (
            f'Você é a {ASSISTANT_NAME}, uma assistente virtual simpática e prestativa. '
            'Seja concisa e natural. Português Brasil. '
            'Responda SEM usar emojis, asteriscos, parênteses ou caracteres especiais '
            'na sua resposta. Mantenha respostas claras e diretas (máximo 2-3 frases).'
        )
        if memory_context:
            system_prompt += f' {memory_context}'
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # Histórico recente (limitado para melhor performance)
        for msg in self.conversation_history[-3:]:  # Reduzido de 4 para 3
            messages.append(msg)
        
        messages.append({'role': 'user', 'content': user_text})
        
        # Obter resposta via LLMClient (Ollama ou LM Studio)
        try:
            response = self.llm.chat(messages)
            
            # Extrair resposta considerando o campo thinking
            ai_reply = self.extract_ai_response(response)
            
            # Limpar resposta para display
            clean_display = self.clean_text_for_display(ai_reply)
            
            # Dividir a resposta se for muito longa
            max_line_length = 100
            if len(clean_display) > max_line_length:
                words = clean_display.split()
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= max_line_length:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                print(Fore.GREEN + f"\n🤖 {ASSISTANT_NAME}:")
                for line in lines:
                    print(Fore.GREEN + f"  {line}")
            else:
                print(Fore.GREEN + f"\n🤖 {ASSISTANT_NAME}: {clean_display}")
            
        except (LLMError, Exception) as e:
            print(Fore.RED + f"Erro na IA: {e}")
            ai_reply = "Desculpe, tive um problema ao processar."
            return
        
        # 5. Atualizar histórico
        self.conversation_history.append({'role': 'user', 'content': user_text})
        self.conversation_history.append({'role': 'assistant', 'content': ai_reply})
        
        # 6. Limpar texto para TTS
        clean_for_tts = self.clean_text_for_tts(ai_reply)
        
        # 7. Converter para áudio e reproduzir
        if clean_for_tts:
            self.reset_inactivity_counter()
            audio_file = self.text_to_speech(clean_for_tts)
            if audio_file:
                self.play_audio_with_interruption(audio_file)
        
        # 8. Atualizar contador de inatividade após resposta
        self.reset_inactivity_counter()

        # 9. Extrair memórias periodicamente (a cada 5 interações)
        self._conversation_count += 1
        if self._conversation_count % 5 == 0:
            self._extract_memory(user_text, ai_reply)

    def text_to_speech(self, text):
        """Converte texto para áudio"""
        if not text:
            return None
        
        try:
            # Dividir texto em frases menores para processamento mais rápido
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            audio_chunks = []
            
            # Processar cada frase separadamente para melhor performance
            for sentence in sentences:
                if not sentence:
                    continue
                    
                # Adicionar pontuação final se necessário
                if not sentence[-1] in '.!?':
                    sentence += '.'
                
                # Usar sistema TTS baseado na configuração
                if self.tts_system == 'kokoro':
                    # Processar configuração de voz kokoro
                    voice_config = parse_voice_config(TTS_VOICE)
                    
                    # Verificar se é mesclagem de vozes
                    if len(voice_config) == 1:
                        # Voz única
                        voice_name = list(voice_config.keys())[0]
                        generator = self.tts_pipeline(sentence, voice=voice_name, speed=TTS_SPEED)
                        
                        for _, _, audio in generator:
                            audio_chunks.append(audio)
                            # Limitar para evitar muito processamento
                            if len(audio_chunks) > 30:
                                break
                    else:
                        # Mesclagem de vozes
                        voices = list(voice_config.keys())
                        percents = list(voice_config.values())
                        
                        # Gerar áudio para cada voz
                        voice_audios = []
                        for voice_name in voices:
                            generator = self.tts_pipeline(sentence, voice=voice_name, speed=TTS_SPEED)
                            voice_audio_chunks = []
                            for _, _, audio in generator:
                                voice_audio_chunks.append(audio)
                                if len(voice_audio_chunks) > 10:  # Limitar para performance
                                    break
                            
                            if voice_audio_chunks:
                                # Concatenar chunks da voz atual
                                voice_audio = np.concatenate(voice_audio_chunks)
                                voice_audios.append(voice_audio)
                        
                        # Mesclar as vozes se tivermos pelo menos uma
                        if voice_audios:
                            # Garantir que todos os áudios tenham o mesmo comprimento
                            min_length = min(len(audio) for audio in voice_audios)
                            trimmed_audios = [audio[:min_length] for audio in voice_audios]
                            
                            # Aplicar pesos das vozes
                            weighted_audios = []
                            for i, audio in enumerate(trimmed_audios):
                                weight = percents[i] / 100.0
                                weighted_audios.append(audio * weight)
                            
                            # Combinar as vozes
                            mixed_audio = np.sum(weighted_audios, axis=0)
                            
                            # Normalizar para evitar clipping
                            max_val = np.max(np.abs(mixed_audio))
                            if max_val > 1.0:
                                mixed_audio = mixed_audio / max_val * 0.95
                            
                            audio_chunks.append(mixed_audio)
                else:
                    # Usar Qwen3-TTS
                    if self.qwen3_pipeline:
                        # Gerar áudio com Qwen3-TTS
                        try:
                            # Verificar cache para frases curtas (melhora performance)
                            cache_key = f"{sentence}_{QWEN3_VOICE}_{QWEN3_LANGUAGE}"
                            if len(sentence) < 100 and cache_key in self.tts_cache:
                                # Usar áudio do cache
                                audio = self.tts_cache[cache_key]
                                audio_chunks.append(audio)
                                print(Fore.CYAN + f"   • Cache hit: '{sentence[:50]}...'")
                            else:
                                # Gerar novo áudio
                                result = self.qwen3_pipeline.generate_custom_voice(
                                    text=sentence,
                                    speaker=QWEN3_VOICE,
                                    language=QWEN3_LANGUAGE,
                                    non_streaming_mode=True
                                )
                                
                                if result and len(result) > 0:
                                    # O resultado é uma tupla (audio_list, sample_rate)
                                    audio_list, sample_rate = result
                                    if audio_list and len(audio_list) > 0:
                                        audio = audio_list[0]  # Pegar o primeiro áudio
                                        audio_chunks.append(audio)
                                        
                                        # Armazenar no cache para frases curtas
                                        if len(sentence) < 100:
                                            # Limpar cache se estiver muito grande
                                            if len(self.tts_cache) >= self.max_cache_size:
                                                # Remover item mais antigo (simples)
                                                first_key = next(iter(self.tts_cache))
                                                del self.tts_cache[first_key]
                                            self.tts_cache[cache_key] = audio
                                            print(Fore.CYAN + f"   • Cache stored: '{sentence[:50]}...'")
                        except Exception as e:
                            print(Fore.YELLOW + f"⚠️  Erro ao gerar áudio com Qwen3-TTS: {e}")
                            # Tentar fallback para Kokoro-TTS se disponível
                            if self.tts_pipeline:
                                print(Fore.YELLOW + f"⚠️  Tentando fallback para Kokoro-TTS...")
                                voice_config = parse_voice_config(TTS_VOICE)
                                if len(voice_config) == 1:
                                    voice_name = list(voice_config.keys())[0]
                                    generator = self.tts_pipeline(sentence, voice=voice_name, speed=TTS_SPEED)
                                    for _, _, audio in generator:
                                        audio_chunks.append(audio)
                                        break
            
            if audio_chunks:
                # Converter chunks de tensor para numpy array antes de processar
                audio_chunks_np = []
                for chunk in audio_chunks:
                    # Converter tensor para numpy se necessário
                    if hasattr(chunk, 'numpy'):
                        audio_chunks_np.append(chunk.numpy())
                    else:
                        audio_chunks_np.append(chunk)
                
                audio_chunks = audio_chunks_np
                
                # Suavizar transições entre chunks
                if len(audio_chunks) > 1:
                    # Aplicar fade in/out suave entre chunks
                    fade_duration = int(0.02 * TTS_SAMPLE_RATE)  # 20ms fade
                    for i in range(1, len(audio_chunks)):
                        if len(audio_chunks[i-1]) > fade_duration and len(audio_chunks[i]) > fade_duration:
                            # Fade out no final do chunk anterior
                            fade_out = np.linspace(1, 0, fade_duration)
                            faded_end = np.multiply(audio_chunks[i-1][-fade_duration:], fade_out)
                            audio_chunks[i-1] = np.concatenate([audio_chunks[i-1][:-fade_duration], faded_end])
                            
                            # Fade in no início do chunk atual
                            fade_in = np.linspace(0, 1, fade_duration)
                            faded_start = np.multiply(audio_chunks[i][:fade_duration], fade_in)
                            audio_chunks[i] = np.concatenate([faded_start, audio_chunks[i][fade_duration:]])
                
                final_audio = np.concatenate(audio_chunks)
                
                # Salvar em arquivo temporário
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, final_audio, TTS_SAMPLE_RATE)
                
                return temp_file.name
                
        except Exception as e:
            print(Fore.RED + f"Erro no TTS ({self.tts_system}): {e}")
        
        return None
    
    def start_avatar(self):
        """Inicia o avatar"""
        if not config.AVATAR_ENABLE:
            print(Fore.YELLOW + "⚠️  Avatar desabilitado, ignorando start_avatar()")
            return
        
        if not self.avatar_started:
            if self.avatar.init_window():
                self.avatar.start()
                self.avatar_started = True
                print(Fore.GREEN + "✅ Avatar pronto para animação")
            else:
                print(Fore.YELLOW + "⚠️  Não foi possível iniciar o avatar. Continuando sem animação.")
    
    def update_avatar(self):
        """Atualiza o avatar (deve ser chamado periodicamente)"""
        if not config.AVATAR_ENABLE:
            return False
        
        try:
            if self.avatar_started and self.avatar:
                return self.avatar.update_and_render()
            elif self.avatar and not self.avatar_started:
                # Tentar iniciar o avatar se ainda não foi iniciado
                self.start_avatar()
                if self.avatar_started:
                    return self.avatar.update_and_render()
            return False
        
        except Exception as e:
            print(Fore.YELLOW + f"⚠️  Erro ao atualizar avatar: {e}")
            return False
    
    def play_audio_with_interruption(self, audio_path):
        """Reproduz áudio com possibilidade de interrupção"""
        if not audio_path or not os.path.exists(audio_path):
            return
        
        # Iniciar avatar se ainda não foi iniciado
        if config.AVATAR_ENABLE:
            self.start_avatar()
        
        try:
            # Ler áudio e converter para float32 (compatível com sounddevice)
            audio_data, samplerate = sf.read(audio_path, dtype='float32')
            
            # Garantir que o áudio está no formato correto
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ativar modo de fala da IA
            self.is_speaking_tts = True
            self.interruption_buffer = []  # Limpar buffer de interrupção
            
            # Ativar animação de fala no avatar
            if config.AVATAR_ENABLE and self.avatar_started:
                self.avatar.set_speaking(True)
            
            # Usar threading para verificação de interrupção em paralelo
            interruption_detected = False
            
            def check_interruption_thread():
                nonlocal interruption_detected
                while not interruption_detected and self.is_speaking_tts:
                    if self.check_interruption():
                        interruption_detected = True
                        break
                    time.sleep(0.05)  # Verificar a cada 50ms (mais responsivo)
            
            # Iniciar thread de verificação de interrupção
            interruption_thread = threading.Thread(target=check_interruption_thread, daemon=True)
            interruption_thread.start()
            
            # Reproduzir áudio de forma contínua com buffer streaming
            # para evitar cortes entre chunks
            stream = sd.OutputStream(
                samplerate=samplerate,
                channels=1,
                dtype='float32',
                blocksize=1024  # Tamanho menor do bloco para menor latência
            )
            
            with stream:
                total_samples = len(audio_data)
                position = 0
                chunk_size = 2048  # Pequenos chunks para streaming suave
                
                while position < total_samples and not interruption_detected:
                    chunk = audio_data[position:min(position + chunk_size, total_samples)]
                    stream.write(chunk)
                    position += len(chunk)
                    
                    # Pequena pausa para permitir verificação de interrupção
                    time.sleep(0.01)
            
            # Sinalizar para a thread parar
            interruption_detected = True
            
            # Esperar thread terminar
            interruption_thread.join(timeout=0.5)
            
            if interruption_detected:
                print(Fore.YELLOW + f"\n🛑 {ASSISTANT_NAME} interrompida pelo usuário!")
            
            # Desativar modo de fala
            self.is_speaking_tts = False
            self.interruption_buffer = []
            
            # Desativar animação de fala no avatar
            if config.AVATAR_ENABLE and self.avatar_started:
                self.avatar.set_speaking(False)
            
            # Limpar arquivo
            try:
                os.unlink(audio_path)
            except:
                pass
                
        except Exception as e:
            print(Fore.RED + f"\nErro ao reproduzir áudio: {e}")
            self.is_speaking_tts = False
            # Desativar animação de fala no avatar em caso de erro
            if config.AVATAR_ENABLE and self.avatar_started:
                self.avatar.set_speaking(False)
    
    def play_audio(self, audio_path):
        """Reproduz áudio normalmente (para saudação inicial)"""
        if not audio_path or not os.path.exists(audio_path):
            return
        
        try:
            audio_data, samplerate = sf.read(audio_path)
            sd.play(audio_data, samplerate)
            sd.wait()
            
            # Limpar arquivo
            try:
                os.unlink(audio_path)
            except:
                pass
                
        except Exception as e:
            print(Fore.RED + f"\nErro ao reproduzir áudio: {e}")
    
    def run(self):
        """Executa o chat contínuo"""
        print(Fore.YELLOW + "\n🎯 MODO POR VOZ ATIVADO")
        print(Fore.CYAN + "\nInstruções:")
        print(f"• Diga '{WAKE_WORDS[0]}' para acordar {ASSISTANT_NAME} (ou outras variações)")
        print(f"• Fale naturalmente - ela detecta pausas automaticamente")
        print(f"• Após {INACTIVITY_TIMEOUT}s sem falar, ela volta a dormir")
        print(f"• Diga 'calado', 'calada' ou 'silêncio' para interrompê-la")
        print(f"• Sensibilidade: {SPEECH_THRESHOLD} (ajuste nas linhas 48-62)")
        print("• Pressione CTRL+C para sair")
        print(Fore.CYAN + "-"*40)
        print(Fore.GREEN + f"\n🎤 {ASSISTANT_NAME} pronta para ouvir...")
        
        # Configurar stream de áudio
        stream_kwargs = {
            'samplerate': SAMPLE_RATE,
            'channels': CHANNELS,
            'dtype': 'float32',
            'blocksize': CHUNK,
            'callback': self.audio_callback
        }
        
        # Adicionar dispositivo se especificado
        if self.audio_device_id is not None:
            stream_kwargs['device'] = self.audio_device_id
            print(Fore.GREEN + f"🎤 Usando dispositivo de áudio: {AUDIO_DEVICE}")
        
        stream = sd.InputStream(**stream_kwargs)
        
        try:
            with stream:
                last_status_check = time.time()
                last_avatar_update = time.time()
                avatar_update_interval = 0.016  # ~60 FPS
                
                while self.is_listening:
                    current_time = time.time()
                    
                    # Verificar inatividade a cada 5 segundos
                    if current_time - last_status_check > 5.0:
                        self.check_inactivity()
                        last_status_check = current_time
                    
                    # Atualizar avatar periodicamente
                    if current_time - last_avatar_update > avatar_update_interval:
                        self.update_avatar()
                        last_avatar_update = current_time
                    
                    time.sleep(0.001)  # Sleep mais curto para melhor responsividade
                    
        except Exception as e:
            print(Fore.RED + f"\nErro no stream de áudio: {e}")
        
        print(Fore.CYAN + "\n" + "="*60)
        print(Fore.GREEN + f"{ASSISTANT_NAME} encerrada.")
        print(Fore.CYAN + "="*60)

def main():
    """Função principal"""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print(Fore.RED + "Erro: Instale as dependências necessárias:")
        print("pip install sounddevice soundfile colorama")
        return
    
    # Verificar se o provedor LLM está rodando
    if LLM_PROVIDER == 'ollama':
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code != 200:
                print(Fore.YELLOW + "⚠️  Ollama não está respondendo. Certifique-se de que está rodando:")
                print("  ollama serve")
                print("\nContinuando em 3 segundos...")
                time.sleep(3)
        except:
            print(Fore.YELLOW + "⚠️  Não foi possível conectar ao Ollama local.")
            print("  Execute: ollama serve")
            print("\nContinuando em 3 segundos...")
            time.sleep(3)
    else:
        # LM Studio - verificação rápida
        try:
            import requests
            resp = requests.get(f'http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/models', timeout=2)
            if resp.status_code != 200:
                print(Fore.YELLOW + f"⚠️  LM Studio ({LM_STUDIO_HOST}:{LM_STUDIO_PORT}) não respondeu corretamente.")
                print(f"  Certifique-se de que o servidor LM Studio está rodando e a API está habilitada.")
                print("\nContinuando em 3 segundos...")
                time.sleep(3)
        except:
            print(Fore.YELLOW + f"⚠️  Não foi possível conectar ao LM Studio em {LM_STUDIO_HOST}:{LM_STUDIO_PORT}.")
            print("  Certifique-se de que:")
            print("  • O LM Studio está rodando")
            print("  • A API está habilitada (Settings > Enable API)")
            print("\nContinuando em 3 segundos...")
            time.sleep(3)
    
    # Criar e executar assistente
    chica = ChicaAssistant()
    
    try:
        chica.run()
    except KeyboardInterrupt:
        print(Fore.RED + "\n\n🛑 Interrompido pelo usuário.")
    except Exception as e:
        print(Fore.RED + f"\nErro fatal: {e}")
    
    print(Fore.CYAN + f"\nAté logo! {ASSISTANT_NAME} 👋")

if __name__ == "__main__":
    main()