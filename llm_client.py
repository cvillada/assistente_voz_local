#!/usr/bin/env python3
"""
Cliente LLM abstrato — suporta Ollama, LM Studio e llama.cpp.

Uso:
    from llm_client import LLMClient

    client = LLMClient(provider='ollama', model='qwen3:1.7b')
    # ou client = LLMClient(provider='lm_studio', model='llama-3.2-3b-instruct', lm_studio_port=1234)
    # ou client = LLMClient(provider='llamacpp', model='qwen2.5-3b-instruct', llamacpp_port=8080)
    response = client.chat([{'role': 'user', 'content': 'Olá!'}])
    print(response.message.content)           # Texto da resposta
    print(response.message.thinking or '')    # Thinking (apenas Ollama)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classes de resposta compatíveis com o formato do Ollama
# ---------------------------------------------------------------------------

@dataclass
class LLMMessage:
    """Mensagem de resposta, compatível com response.message do Ollama."""
    content: str = ""
    thinking: Optional[str] = None


@dataclass
class LLMResponse:
    """Resposta completa, compatível com o objeto response do Ollama."""
    message: LLMMessage = field(default_factory=LLMMessage)


# ---------------------------------------------------------------------------
# Exceção personalizada
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Erro genérico do provedor LLM."""
    pass


# ---------------------------------------------------------------------------
# Cliente principal
# ---------------------------------------------------------------------------

class LLMClient:
    """Cliente unificado para Ollama e LM Studio.

    Args:
        provider: 'ollama', 'lm_studio' ou 'llamacpp'
        model: Nome do modelo (ex: 'qwen3:1.7b' para Ollama,
               'llama-3.2-3b-instruct' para LM Studio/llama.cpp)
        temperature: Criatividade (0.0 - 1.0)
        max_tokens: Máximo de tokens na resposta
        lm_studio_host: Host do LM Studio (padrão: 'localhost')
        lm_studio_port: Porta do LM Studio (padrão: 1234)
        llamacpp_host: Host do llama-server (padrão: 'localhost')
        llamacpp_port: Porta do llama-server (padrão: 8080)
        request_timeout: Timeout em segundos para cada requisição
    """

    def __init__(
        self,
        provider: str = 'ollama',
        model: str = 'qwen3:1.7b',
        temperature: float = 0.7,
        max_tokens: int = 300,
        lm_studio_host: str = 'localhost',
        lm_studio_port: int = 1234,
        llamacpp_host: str = 'localhost',
        llamacpp_port: int = 8080,
        request_timeout: int = 60,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lm_studio_base = f"http://{lm_studio_host}:{lm_studio_port}"
        self.llamacpp_base = f"http://{llamacpp_host}:{llamacpp_port}"
        self.request_timeout = request_timeout
        self._backend = None

        if self.provider not in ('ollama', 'lm_studio', 'llamacpp'):
            raise LLMError(
                f"Provedor desconhecido: '{provider}'. Use 'ollama', 'lm_studio' ou 'llamacpp'."
            )

    # ------------------------------------------------------------------
    # Inicialização sob demanda (lazy)
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> None:
        """Importa e prepara o backend na primeira chamada."""
        if self._backend is not None:
            return

        if self.provider == 'ollama':
            self._init_ollama()
        elif self.provider == 'lm_studio':
            self._init_lm_studio()
        else:
            self._init_llamacpp()

    def _init_ollama(self) -> None:
        try:
            import ollama  # type: ignore[import-untyped]
            self._backend = ollama
            logger.info("Backend Ollama inicializado")
        except ImportError:
            raise LLMError(
                "Pacote 'ollama' não encontrado. Instale com: pip install ollama"
            )

    def _init_lm_studio(self) -> None:
        # Tenta openai primeiro (recomendado), fallback para requests
        try:
            from openai import OpenAI
            self._backend = OpenAI(
                base_url=f"{self.lm_studio_base}/v1",
                api_key="lm-studio",  # LM Studio aceita qualquer valor
                timeout=self.request_timeout,
            )
            self._backend._is_openai = True
            logger.info(
                "Backend LM Studio inicializado via openai library"
            )
        except ImportError:
            try:
                import requests as req
                self._backend = req
                self._backend._is_openai = False
                logger.info(
                    "Backend LM Studio inicializado via requests (fallback)"
                )
            except ImportError:
                raise LLMError(
                    "Pacote 'openai' ou 'requests' necessário para LM Studio. "
                    "Instale com: pip install openai"
                )

    def _init_llamacpp(self) -> None:
        """Inicializa backend para llama.cpp (mesma API OpenAI-compatible)."""
        try:
            from openai import OpenAI
            self._backend = OpenAI(
                base_url=f"{self.llamacpp_base}/v1",
                api_key="not-needed",  # llama.cpp não requer API key
                timeout=self.request_timeout,
            )
            self._backend._is_openai = True
            logger.info(
                "Backend llama.cpp inicializado via openai library"
            )
        except ImportError:
            try:
                import requests as req
                self._backend = req
                self._backend._is_openai = False
                logger.info(
                    "Backend llama.cpp inicializado via requests (fallback)"
                )
            except ImportError:
                raise LLMError(
                    "Pacote 'openai' ou 'requests' necessário para llama.cpp. "
                    "Instale com: pip install openai"
                )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict]) -> LLMResponse:
        """Envia mensagens para o modelo e retorna a resposta."""
        self._ensure_backend()

        if self.provider == 'ollama':
            return self._chat_ollama(messages)
        else:
            # LM Studio e llama.cpp usam a mesma API OpenAI-compatible
            return self._chat_lm_studio(messages)

    def _chat_ollama(self, messages: list[dict]) -> LLMResponse:
        try:
            response = self._backend.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                },
            )
            msg = response.message
            return LLMResponse(
                message=LLMMessage(
                    content=getattr(msg, 'content', '') or '',
                    thinking=getattr(msg, 'thinking', None),
                )
            )
        except Exception as e:
            raise LLMError(f"Erro no Ollama: {e}")

    def _chat_lm_studio(self, messages: list[dict]) -> LLMResponse:
        if getattr(self._backend, '_is_openai', False):
            return self._chat_lm_studio_openai(messages)
        else:
            return self._chat_lm_studio_requests(messages)

    def _chat_lm_studio_openai(self, messages: list[dict]) -> LLMResponse:
        try:
            response = self._backend.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            choice = response.choices[0]
            content = choice.message.content or ''
            return LLMResponse(
                message=LLMMessage(content=content, thinking=None)
            )
        except Exception as e:
            raise LLMError(f"Erro no LM Studio (OpenAI): {e}")

    def _chat_lm_studio_requests(self, messages: list[dict]) -> LLMResponse:
        try:
            base_url = self.llamacpp_base if self.provider == 'llamacpp' else self.lm_studio_base
            resp = self._backend.post(
                f"{base_url}/v1/chat/completions",
                json={
                    'model': self.model,
                    'messages': messages,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                },
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data['choices'][0]['message']['content'] or ''
            return LLMResponse(
                message=LLMMessage(content=content, thinking=None)
            )
        except Exception as e:
            raise LLMError(f"Erro no LM Studio (requests): {e}")

    # ------------------------------------------------------------------
    # Chat com streaming (para uso futuro)
    # ------------------------------------------------------------------

    def chat_stream(self, messages: list[dict]):
        """Versão streaming (apenas Ollama, LM Studio e llama.cpp via openai library)."""
        self._ensure_backend()

        if self.provider == 'ollama':
            return self._backend.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                },
            )
        else:
            # LM Studio e llama.cpp via OpenAI library suportam streaming
            if getattr(self._backend, '_is_openai', False):
                stream = self._backend.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                return stream
            raise LLMError("LM Studio/llama.cpp via requests não suporta streaming")

    # ------------------------------------------------------------------
    # Verificação de disponibilidade
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Verifica se o provedor está respondendo."""
        self._ensure_backend()

        if self.provider == 'ollama':
            return self._is_ollama_available()
        else:
            # LM Studio e llama.cpp usam o mesmo endpoint /v1/models
            return self._is_lm_studio_available()

    def _is_ollama_available(self) -> bool:
        try:
            self._backend.list()  # ollama.list() não lança exceção se servidor OK
            return True
        except Exception:
            return False

    def _is_lm_studio_available(self) -> bool:
        if getattr(self._backend, '_is_openai', False):
            try:
                self._backend.models.list()
                return True
            except Exception:
                return False
        else:
            try:
                base_url = self.llamacpp_base if self.provider == 'llamacpp' else self.lm_studio_base
                resp = self._backend.get(
                    f"{base_url}/v1/models",
                    timeout=5,
                )
                return resp.status_code == 200
            except Exception:
                return False
