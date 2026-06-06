#!/usr/bin/env python3
"""
Gerenciador de memória persistente para a assistente Chica.

Armazena fatos importantes sobre o usuário em um arquivo JSON.
As memórias são injetadas no system prompt para dar contexto
às conversas futuras.

Leve o suficiente para Raspberry Pi 4 — sem banco de dados,
sem embeddings, sem busca vetorial.

Uso:
    from memory_manager import MemoryManager

    mem = MemoryManager("memories.json")
    mem.add("Usuário se chama Claudinei")
    mem.add("Usuário prefere respostas curtas")
    ctx = mem.get_context()
    # ctx -> "📝 Memórias: Usuário se chama Claudinei | Usuário prefere respostas curtas"
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from log import logger


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_MEMORIES = 100          # Máximo de memórias armazenadas
MAX_AGE_DAYS = 30          # Máximo de dias antes de expirar
MAX_FACT_LENGTH = 120      # Máximo de caracteres por fato


# ---------------------------------------------------------------------------
# Gerenciador
# ---------------------------------------------------------------------------

class MemoryManager:
    """Gerencia memórias persistentes em arquivo JSON."""

    def __init__(self, filepath: str) -> None:
        self.filepath: str = os.path.abspath(filepath)
        self._memories: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Carrega memórias do arquivo JSON."""
        if not os.path.exists(self.filepath):
            self._memories = []
            logger.info("Arquivo de memória não encontrado. Iniciando vazio.")
            return
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            self._memories = data if isinstance(data, list) else []
            # Limpar memórias expiradas
            self._prune_expired()
            logger.success(f"{len(self._memories)} memórias carregadas")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Erro ao carregar memórias: {e}. Iniciando vazio.")
            self._memories = []

    def save(self) -> None:
        """Salva memórias no arquivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self._memories, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.warning(f"Erro ao salvar memórias: {e}")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, fact: str) -> None:
        """Adiciona um fato à memória.

        Remove duplicatas (pelo texto) e mantém o máximo de MAX_MEMORIES.
        """
        fact = fact.strip()
        if not fact or len(fact) > MAX_FACT_LENGTH:
            return

        # Remover duplicata se existir (atualiza timestamp)
        for m in self._memories:
            if m['text'].lower() == fact.lower():
                m['timestamp'] = time.time()
                m['access_count'] = m.get('access_count', 0) + 1
                self.save()
                return

        # Adicionar nova
        self._memories.append({
            'text': fact,
            'timestamp': time.time(),
            'access_count': 1,
        })

        # Podar se excedeu o limite
        self._prune()
        self.save()
        logger.info(f"🧠 Memória salva: {fact[:60]}...")

    def remove(self, text: str) -> None:
        """Remove uma memória pelo texto (busca parcial)."""
        before = len(self._memories)
        self._memories = [
            m for m in self._memories
            if text.lower() not in m['text'].lower()
        ]
        if len(self._memories) < before:
            self.save()
            logger.info(f"🗑️ Memória removida: {text[:60]}...")

    def clear(self) -> None:
        """Limpa todas as memórias."""
        self._memories = []
        self.save()
        logger.info("🧹 Todas as memórias foram limpas")

    # ------------------------------------------------------------------
    # Contexto para o LLM
    # ------------------------------------------------------------------

    def get_context(self) -> str:
        """Retorna as memórias formatadas para injeção no system prompt.

        Retorna string vazia se não houver memórias.
        Ordena por access_count (mais acessadas primeiro).
        """
        if not self._memories:
            return ""

        # Ordenar: mais acessadas primeiro, depois mais recentes
        sorted_m = sorted(
            self._memories,
            key=lambda m: (m.get('access_count', 0), m.get('timestamp', 0)),
            reverse=True,
        )

        facts = [m['text'] for m in sorted_m[:MAX_MEMORIES]]
        return "📝 Memórias sobre o usuário: " + " | ".join(facts)

    def count(self) -> int:
        """Número de memórias atuais."""
        return len(self._memories)

    # ------------------------------------------------------------------
    # Manutenção interna
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove o excesso de memórias (as menos acessadas/mais antigas)."""
        if len(self._memories) <= MAX_MEMORIES:
            return

        # Ordenar: menos acessadas primeiro, depois mais antigas
        self._memories.sort(
            key=lambda m: (m.get('access_count', 0), m.get('timestamp', 0)),
        )

        # Remover as piores até caber
        self._memories = self._memories[-MAX_MEMORIES:]

    def _prune_expired(self) -> None:
        """Remove memórias mais velhas que MAX_AGE_DAYS."""
        cutoff = time.time() - (MAX_AGE_DAYS * 86400)
        before = len(self._memories)
        self._memories = [
            m for m in self._memories
            if m.get('timestamp', 0) > cutoff
        ]
        if len(self._memories) < before:
            logger.info(f"🧹 {before - len(self._memories)} memórias expiradas removidas")
