#!/usr/bin/env python3
"""
Detector de voz em tempo real — captura áudio, detecta fala,
gerencia wake words e interrupções.

Uso:
    from audio_detector import AudioDetector

    detector = AudioDetector()
    detector.load_stt_model('base')
    detector.set_callbacks(on_speech=minha_funcao)
    detector.start()
"""

from __future__ import annotations

import os
import re
import string
import tempfile
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

import config
from log import logger


CallbackType = Callable[[str], None]


class AudioDetector:
    """Captura e processa áudio em tempo real com detecção de fala."""

    def __init__(self) -> None:
        # Áudio
        self.sample_rate: int = config.SAMPLE_RATE
        self.channels: int = config.CHANNELS
        self.chunk: int = config.CHUNK
        self.audio_device_id: Optional[int] = self._resolve_device()

        # Detecção adaptativa
        self.noise_floor: float = config.INITIAL_NOISE_FLOOR
        self.speech_threshold: float = config.SPEECH_THRESHOLD

        # Estado
        self.audio_buffer: list[np.ndarray] = []
        self.interruption_buffer: list[np.ndarray] = []
        self.user_is_speaking: bool = False
        self.is_processing: bool = False
        self.is_speaking_tts: bool = False
        self.is_active: bool = False
        self.last_speech_time: float = time.time()
        self.last_activity_time: float = time.time()
        self.consecutive_speech_chunks: int = 0
        self.silence_chunks_counter: int = 0
        self.silence_chunks_needed: int = int(
            config.SILENCE_DURATION * config.SAMPLE_RATE / config.CHUNK
        )

        # Callbacks
        self._on_speech_detected: Optional[CallbackType] = None
        self._on_inactivity: Optional[Callable[[], None]] = None
        self._stream: Optional[sd.InputStream] = None

        # STT (faster-whisper)
        self.stt_model: Optional['WhisperModel'] = None

    # ------------------------------------------------------------------
    # Dispositivo de áudio
    # ------------------------------------------------------------------

    def _resolve_device(self) -> Optional[int]:
        """Encontra o ID do dispositivo de áudio baseado na config."""
        device_name = config.AUDIO_DEVICE.lower()
        if device_name in ("padrão", "default"):
            return None
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                dn = dev['name'].lower()
                if any(x in dn for x in ['isolamento', 'voice isolation', 'voice_isolation']):
                    logger.success(f"Dispositivo encontrado: {dev['name']}")
                    return i
            for i, dev in enumerate(devices):
                if device_name in dev['name'].lower():
                    logger.success(f"Dispositivo: {dev['name']}")
                    return i
            logger.warning(f"Dispositivo '{config.AUDIO_DEVICE}' não encontrado. Usando padrão.")
        except Exception as e:
            logger.warning(f"Erro ao listar dispositivos: {e}")
        return None

    # ------------------------------------------------------------------
    # Modelo STT (Whisper)
    # ------------------------------------------------------------------

    def load_stt_model(self, model_name: str = 'base') -> None:
        """Carrega o modelo Whisper (faster-whisper) para transcrição."""
        hf_name = "large-v3-turbo" if model_name == "turbo" else model_name
        try:
            self.stt_model = WhisperModel(
                hf_name, device="cpu", compute_type="int8",
                cpu_threads=4, num_workers=2,
            )
            logger.success(f"faster-whisper '{model_name}' carregado (int8)")
        except Exception:
            logger.warning(f"Falha ao carregar '{model_name}', tentando 'tiny'...")
            try:
                self.stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                logger.success("faster-whisper 'tiny' carregado como fallback")
            except Exception as e:
                logger.error(f"Erro crítico ao carregar modelo Whisper: {e}")

    def transcribe(self, audio_path: str) -> str:
        """Transcreve um arquivo de áudio para texto (faster-whisper)."""
        if not self.stt_model:
            return ""
        try:
            segments, _info = self.stt_model.transcribe(
                audio_path,
                language=config.WHISPER_LANGUAGE,
                beam_size=3,
                vad_filter=True,
            )
            return " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            logger.error(f"Erro na transcrição: {e}")
            return ""

    # ------------------------------------------------------------------
    # Wake words e comandos
    # ------------------------------------------------------------------

    def check_wake_word(self, text: str) -> bool:
        """Verifica se o texto contém uma wake word."""
        text_lower = text.lower().strip()
        text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
        words = text_clean.split()
        words_need_context = ["shika", "shica", "chica"]
        prefixes = ["olá", "ola", "oi", "ei", "hey", "hei", "ok", "okay", "tá", "ta", "pronto"]

        for wake_word in config.WAKE_WORDS:
            search_in = text_lower if wake_word in text_lower else ""
            search_in_clean = text_clean if wake_word in text_clean else ""
            if not (search_in or search_in_clean):
                continue
            if any(w in wake_word.split() for w in words_need_context):
                parts = wake_word.split()
                for i in range(len(words) - len(parts) + 1):
                    if words[i:i+len(parts)] == parts:
                        if i == 0 or (i > 0 and words[i-1] in prefixes):
                            return True
            else:
                return True

        # Variações sem acentos
        for wake_word in config.WAKE_WORDS:
            no_acc = wake_word.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
            if no_acc not in text_lower and no_acc not in text_clean:
                continue
            if any(w in no_acc.split() for w in words_need_context):
                parts = no_acc.split()
                for i in range(len(words) - len(parts) + 1):
                    if words[i:i+len(parts)] == parts and (i == 0 or words[i-1] in prefixes):
                        return True
            else:
                return True
        return False

    def check_stop_command(self, text: str) -> bool:
        """Verifica se o texto contém comando para parar a fala."""
        text_lower = text.lower().strip()
        # Busca por palavra inteira (evita "para" em "parabéns")
        for phrase in config.STOP_PHRASES:
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                return True
        words = text_lower.split()
        if len(words) >= 2 and config.ASSISTANT_NAME.lower() in words[0] and words[1] in config.STOP_PHRASES:
            return True
        return False

    # ------------------------------------------------------------------
    # Callback de áudio (chamado pelo sounddevice)
    # ------------------------------------------------------------------

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback do stream de áudio — processa chunks em tempo real."""
        if status:
            return

        if self.is_speaking_tts:
            self.interruption_buffer.append(indata.copy())
            max_size = int(3.0 * self.sample_rate / self.chunk)
            if len(self.interruption_buffer) > max_size:
                self.interruption_buffer = self.interruption_buffer[-max_size:]
            return

        audio_chunk = indata.copy()
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

        if rms < self.speech_threshold * config.NOISE_FLOOR_UPDATE_THRESHOLD:
            self.noise_floor = (
                self.noise_floor * config.NOISE_FLOOR_SMOOTHING
                + rms * (1 - config.NOISE_FLOOR_SMOOTHING)
            )

        dynamic_threshold = max(self.speech_threshold, self.noise_floor * config.DYNAMIC_THRESHOLD_MULTIPLIER)
        is_speech = rms > dynamic_threshold

        if is_speech:
            self.consecutive_speech_chunks += 1
            self.silence_chunks_counter = 0
            if not self.user_is_speaking and self.consecutive_speech_chunks > 1:
                self.user_is_speaking = True
                self.last_speech_time = time.time()
                if self.is_active:
                    self.last_activity_time = time.time()
            self.audio_buffer.append(audio_chunk)
            max_buf = int(10.0 * self.sample_rate / self.chunk)
            if len(self.audio_buffer) > max_buf:
                self.audio_buffer = self.audio_buffer[-max_buf:]
        else:
            self.consecutive_speech_chunks = 0
            self.silence_chunks_counter += 1
            if self.user_is_speaking:
                self.audio_buffer.append(audio_chunk)
                if self.silence_chunks_counter >= self.silence_chunks_needed:
                    self.user_is_speaking = False
                    buf_duration = len(self.audio_buffer) * self.chunk / self.sample_rate
                    if buf_duration >= config.MIN_SPEECH_DURATION and not self.is_processing and self._on_speech_detected:
                        self.is_processing = True
                        threading.Thread(target=self._process_buffer, daemon=True).start()

    def _process_buffer(self) -> None:
        """Processa o buffer de áudio (roda em thread separada)."""
        if not self.audio_buffer or not self._on_speech_detected:
            self.is_processing = False
            return
        try:
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            energy = float(np.sqrt(np.mean(audio_data ** 2)))
            if energy < self.speech_threshold * config.SPEECH_ENERGY_MULTIPLIER:
                self.audio_buffer.clear()
                self.is_processing = False
                return
            self.audio_buffer.clear()
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(tmp.name, audio_data, self.sample_rate)
            text = self.transcribe(tmp.name)
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            if text:
                self._on_speech_detected(text)
        except Exception as e:
            logger.error(f"Erro ao processar áudio: {e}")
        finally:
            self.is_processing = False

    # ------------------------------------------------------------------
    # Interrupção durante fala da IA
    # ------------------------------------------------------------------

    def check_interruption(self) -> bool:
        """Verifica se o usuário disse um comando de parar durante a fala."""
        if not self.interruption_buffer or self.is_processing:
            return False
        try:
            data = np.concatenate(self.interruption_buffer, axis=0)
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(tmp.name, data, self.sample_rate)
            text = self.transcribe(tmp.name)
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            if text and self.check_stop_command(text):
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Inatividade
    # ------------------------------------------------------------------

    def check_inactivity(self) -> bool:
        """Verifica se está inativo e coloca para dormir se necessário."""
        if self.is_processing or self.is_speaking_tts:
            return False
        if self.is_active and (time.time() - self.last_activity_time) > config.INACTIVITY_TIMEOUT:
            logger.info(f"{config.ASSISTANT_NAME} está dormindo... (inatividade)")
            self.is_active = False
            if self._on_inactivity:
                self._on_inactivity()
            return True
        return False

    def reset_activity(self) -> None:
        """Reseta o timer de atividade."""
        self.last_activity_time = time.time()

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia o stream de áudio."""
        kwargs = {
            'samplerate': self.sample_rate,
            'channels': self.channels,
            'dtype': 'float32',
            'blocksize': self.chunk,
            'callback': self.audio_callback,
        }
        if self.audio_device_id is not None:
            kwargs['device'] = self.audio_device_id
            logger.info(f"Usando dispositivo: {config.AUDIO_DEVICE}")
        self._stream = sd.InputStream(**kwargs)
        self._stream.start()
        logger.success("Stream de áudio iniciado")

    def stop(self) -> None:
        """Para o stream de áudio."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Stream de áudio parado")

    def set_callbacks(
        self,
        on_speech: Optional[CallbackType] = None,
        on_inactivity: Optional[Callable[[], None]] = None,
    ) -> None:
        """Define callbacks para detecção de fala e inatividade."""
        self._on_speech_detected = on_speech
        self._on_inactivity = on_inactivity
