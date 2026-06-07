#!/usr/bin/env python3
"""
Motor de síntese de voz (TTS) — Kokoro e Qwen3.

Uso:
    from tts_engine import TTSManager

    tts = TTSManager(system='kokoro')
    audio_path = tts.synthesize("Olá, mundo!")
    if audio_path:
        print(f"Áudio salvo em {audio_path}")
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from colorama import Fore, init as colorama_init
from kokoro import KPipeline

import config
from log import logger

colorama_init(autoreset=True)


class TTSManager:
    """Gerencia a síntese de voz com Kokoro ou Qwen3."""

    def __init__(self, system: str = 'kokoro') -> None:
        self.system: str = system
        self.kokoro_pipeline: Optional[KPipeline] = None
        self.qwen3_pipeline = None
        self.tts_cache: dict[str, np.ndarray] = {}
        self.max_cache_size: int = 50
        self.qwen3_warmed_up: bool = False
        self._init()

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def _init(self) -> None:
        """Inicializa o pipeline TTS conforme o sistema escolhido."""
        if self.system == 'kokoro':
            self._init_kokoro()
        elif self.system == 'edge':
            self._init_edge()
        else:
            self._init_qwen3()

    def _init_kokoro(self) -> None:
        """Inicializa Kokoro-TTS."""
        logger.info("Inicializando sistema TTS Kokoro...")
        self.kokoro_pipeline = KPipeline(lang_code=config.TTS_KOKORO_LANG, repo_id=config.TTS_KOKORO_MODEL)
        logger.success("Sistema TTS Kokoro inicializado")

    def _init_qwen3(self) -> None:
        """Inicializa Qwen3-TTS com fallback para Kokoro."""
        logger.info(f"Inicializando sistema TTS Qwen3 ({config.QWEN3_MODEL})...")
        try:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

            tts_device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Dispositivo TTS: {tts_device}")

            if tts_device == "mps":
                self.qwen3_pipeline = Qwen3TTSModel.from_pretrained(
                    config.QWEN3_MODEL, device_map="mps"
                )
            else:
                self.qwen3_pipeline = Qwen3TTSModel.from_pretrained(config.QWEN3_MODEL)

            actual_device = (
                self.qwen3_pipeline.model.device
                if hasattr(self.qwen3_pipeline.model, 'device')
                else next(self.qwen3_pipeline.model.parameters()).device
            )
            logger.success(f"Qwen3-TTS inicializado (dispositivo: {actual_device})")
            logger.info(f"   • Voz: {config.QWEN3_VOICE}")
            logger.info(f"   • Idioma: {config.QWEN3_LANGUAGE}")

            self._maybe_compile()
            self._maybe_warmup()

        except ImportError:
            logger.error("Qwen3-TTS não instalado. Instale com: pip install qwen-tts")
            self._fallback_to_kokoro()
        except Exception as e:
            logger.warning(f"Erro ao inicializar Qwen3-TTS: {e}")
            logger.info("Tentando novamente...")
            try:
                from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
                tts_device = "mps" if torch.backends.mps.is_available() else "cpu"
                if tts_device == "mps":
                    self.qwen3_pipeline = Qwen3TTSModel.from_pretrained(
                        config.QWEN3_MODEL, device_map="mps"
                    )
                else:
                    self.qwen3_pipeline = Qwen3TTSModel.from_pretrained(config.QWEN3_MODEL)
                self._maybe_compile()
                self._maybe_warmup()
                logger.success("Qwen3-TTS inicializado na segunda tentativa")
            except Exception:
                logger.error("Não foi possível inicializar Qwen3-TTS")
                self._fallback_to_kokoro()

    def _init_edge(self) -> None:
        """Inicializa Edge-TTS (verifica internet, fallback para Kokoro se offline)."""
        logger.info("Inicializando Edge-TTS...")
        try:
            import edge_tts
            logger.info(f"Edge-TTS disponível (voz: {config.EDGE_TTS_VOICE})")
            # Verificar internet de forma genérica
            import socket
            hosts = [('1.1.1.1', 443), ('8.8.8.8', 443)]
            online = False
            for host, port in hosts:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect((host, port))
                    s.close()
                    online = True
                    break
                except (socket.timeout, OSError):
                    continue
            if online:
                logger.success("Edge-TTS inicializado (online)")
            else:
                logger.warning("Sem internet para Edge-TTS — fallback para Kokoro")
                self._fallback_to_kokoro()
        except ImportError:
            logger.warning("edge-tts não instalado. Instale com: pip install edge-tts")
            self._fallback_to_kokoro()

    def _maybe_compile(self) -> None:
        """Aplica torch.compile se configurado."""
        if config.QWEN3_USE_COMPILE and hasattr(torch, 'compile') and self.qwen3_pipeline:
            try:
                logger.info("Compilando modelo com torch.compile...")
                compiled = torch.compile(self.qwen3_pipeline.model, mode="reduce-overhead")
                self.qwen3_pipeline.model = compiled
                logger.success("Modelo compilado")
            except Exception as e:
                logger.warning(f"Aviso na compilação: {e}")

    def _maybe_warmup(self) -> None:
        """Pré-aquece o modelo Qwen3 para reduzir latência na primeira inferência."""
        if not self.qwen3_pipeline:
            return
        try:
            logger.info("Pré-aquecendo modelo Qwen3-TTS...")
            _ = self.qwen3_pipeline.generate_custom_voice(
                text="Olá",
                speaker=config.QWEN3_VOICE,
                language=config.QWEN3_LANGUAGE,
                non_streaming_mode=True,
            )
            self.qwen3_warmed_up = True
            logger.success("Modelo pré-aquecido")
        except Exception as e:
            logger.warning(f"Aviso no pré-aquecimento: {e}")

    def _fallback_to_kokoro(self) -> None:
        """Fallback para Kokoro quando Qwen3 falha."""
        logger.warning("Usando Kokoro-TTS como fallback...")
        self.system = 'kokoro'
        self.kokoro_pipeline = KPipeline(lang_code=config.TTS_KOKORO_LANG, repo_id=config.TTS_KOKORO_MODEL)
        logger.success("Kokoro-TTS ativado como fallback")

    # ------------------------------------------------------------------
    # Síntese
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> Optional[str]:
        """Converte texto em áudio e retorna o caminho do arquivo .wav."""
        if not text:
            return None

        # Edge-TTS sintetiza o texto completo de uma vez (async)
        if self.system == 'edge':
            return self._edge_synthesize_full(text)

        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            audio_chunks: list[np.ndarray] = []

            for sentence in sentences:
                if not sentence:
                    continue
                if sentence[-1] not in '.!?':
                    sentence += '.'
                chunk = self._synthesize_sentence(sentence)
                if chunk is not None:
                    audio_chunks.append(chunk)

            if not audio_chunks:
                return None

            # Converter tensores para numpy
            chunks_np: list[np.ndarray] = []
            for c in audio_chunks:
                chunks_np.append(c.numpy() if hasattr(c, 'numpy') else c)
            audio_chunks = chunks_np

            # Fade entre chunks
            self._apply_crossfade(audio_chunks)

            final_audio = np.concatenate(audio_chunks)

            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, final_audio, config.TTS_SAMPLE_RATE)
            return temp_file.name

        except Exception as e:
            logger.error(f"Erro no TTS ({self.system}): {e}")
            return None

    def _edge_synthesize_full(self, text: str) -> Optional[str]:
        """Sintetiza o texto completo com Edge-TTS (async)."""
        import asyncio
        try:
            import edge_tts
            communicate = edge_tts.Communicate(
                text,
                config.EDGE_TTS_VOICE,
                rate=f"{int((config.EDGE_TTS_SPEED - 1.0) * 100):+d}%"
            )
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Executar a síntese async de forma síncrona
            asyncio.run(communicate.save(temp_path))

            # Converter MP3 para WAV (o app espera WAV)
            import soundfile as sf
            import numpy as np
            data, sr = sf.read(temp_path)
            os.unlink(temp_path)  # Remove MP3

            # Salvar como WAV
            wav_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            sf.write(wav_path, data, sr)
            return wav_path

        except Exception as e:
            logger.warning(f"Erro no Edge-TTS: {e}")
            logger.info("Fallback para Kokoro nesta síntese...")
            if self.kokoro_pipeline:
                self.system = 'kokoro'
                return self.synthesize(text)
            return None

    def _synthesize_sentence(self, sentence: str) -> Optional[np.ndarray]:
        """Sintetiza uma frase com o sistema TTS atual."""
        if self.system == 'kokoro' and self.kokoro_pipeline:
            return self._kokoro_sentence(sentence)
        elif self.system == 'edge':
            return self._edge_sentence(sentence)
        elif self.qwen3_pipeline:
            return self._qwen3_sentence(sentence)
        return None

    def _kokoro_sentence(self, sentence: str) -> Optional[np.ndarray]:
        """Sintetiza uma frase com Kokoro (suporta mesclagem de vozes)."""
        voice_config = self._parse_voice_config(config.TTS_VOICE)
        assert self.kokoro_pipeline is not None

        if len(voice_config) == 1:
            voice_name = list(voice_config.keys())[0]
            chunks = list(self.kokoro_pipeline(sentence, voice=voice_name, speed=config.TTS_SPEED))
            audios = [a for _, _, a in chunks[:30]]
            return np.concatenate(audios) if audios else None
        else:
            # Mesclagem de vozes
            voices = list(voice_config.keys())
            percents = list(voice_config.values())
            voice_audios: list[np.ndarray] = []
            for voice_name in voices:
                chunks = list(self.kokoro_pipeline(sentence, voice=voice_name, speed=config.TTS_SPEED))
                audios = [a for _, _, a in chunks[:10]]
                if audios:
                    voice_audios.append(np.concatenate(audios))
            if not voice_audios:
                return None
            min_len = min(len(a) for a in voice_audios)
            trimmed = [a[:min_len] * (percents[i] / 100.0) for i, a in enumerate(voice_audios)]
            mixed = np.sum(trimmed, axis=0)
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val * 0.95
            return mixed

    def _qwen3_sentence(self, sentence: str) -> Optional[np.ndarray]:
        """Sintetiza uma frase com Qwen3 (com cache para frases curtas)."""
        assert self.qwen3_pipeline is not None
        cache_key = f"{sentence}_{config.QWEN3_VOICE}_{config.QWEN3_LANGUAGE}"
        if len(sentence) < 100 and cache_key in self.tts_cache:
            logger.info(f"Cache hit: '{sentence[:50]}...'")
            return self.tts_cache[cache_key]

        try:
            result = self.qwen3_pipeline.generate_custom_voice(
                text=sentence,
                speaker=config.QWEN3_VOICE,
                language=config.QWEN3_LANGUAGE,
                non_streaming_mode=True,
            )
            if result and len(result) > 0:
                audio_list, _ = result
                if audio_list and len(audio_list) > 0:
                    audio = audio_list[0]
                    if len(sentence) < 100:
                        if len(self.tts_cache) >= self.max_cache_size:
                            self.tts_cache.pop(next(iter(self.tts_cache)))
                        self.tts_cache[cache_key] = audio
                    return audio
        except Exception as e:
            logger.warning(f"Erro no Qwen3-TTS: {e}")
            if self.kokoro_pipeline:
                logger.info("Fallback para Kokoro nesta frase...")
                return self._kokoro_sentence(sentence)
        return None

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_crossfade(chunks: list[np.ndarray], fade_ms: int = 20) -> None:
        """Aplica fade in/out entre chunks de áudio."""
        if len(chunks) <= 1:
            return
        fade_duration = int(0.001 * fade_ms * config.TTS_SAMPLE_RATE)
        for i in range(1, len(chunks)):
            prev, curr = chunks[i - 1], chunks[i]
            if len(prev) > fade_duration and len(curr) > fade_duration:
                fade_out = np.linspace(1, 0, fade_duration)
                prev[-fade_duration:] = np.multiply(prev[-fade_duration:], fade_out)
                fade_in = np.linspace(0, 1, fade_duration)
                curr[:fade_duration] = np.multiply(curr[:fade_duration], fade_in)

    @staticmethod
    def _parse_voice_config(voice_config: str) -> dict[str, int]:
        """Delega para config.parse_voice_config."""
        from config import parse_voice_config as _parse
        return _parse(voice_config)

    @staticmethod
    def clean_for_tts(text: str) -> str:
        """Remove caracteres que o TTS não lê bem."""
        if not text:
            return ""
        # Emojis
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # Caracteres especiais
        for c in '*~_`|•·©®™→←↑↓↔↕⇒⇐⇑⇓∞≠≤≥≈≡≅∀∃∄∅∆∇∈∉∋∌∏∑√∛∜∝∞∟∠∧∨∩∪∫∬∭∮∴∵∶∷∼∽':
            text = text.replace(c, '')
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r'\{[^}]*\}', '', text)
        text = text.replace('"', '').replace("'", '')
        text = text.replace('—', ', ').replace('–', ', ')
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        # Marcadores de lista
        lines = text.split('\n')
        cleaned = [re.sub(r'^[\s]*[•\-*\d\.\)]+[\s]*', '', l).strip() for l in lines if l.strip()]
        text = ' '.join(cleaned)
        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def clean_for_display(text: str) -> str:
        """Limpeza leve para exibição no terminal."""
        if not text:
            return ""
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        return text.replace('*', '').replace('~', '').replace('_', ' ')
