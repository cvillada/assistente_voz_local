#!/usr/bin/env python3
"""
Gerenciador de memória para a assistente Chica.

Implementa o mesmo padrão do Hermes Agent:
  - Dois arquivos: assistant_memory.md e assistant_user.md
  - Entradas separadas por § (section sign)
  - Limites de caracteres com % de uso visível
  - Injeção no system prompt como snapshot congelado

Uso:
    from memory_manager import MemoryManager
    mem = MemoryManager('.')
    mem.add('user', 'Nome do usuário: Claudinei')
    ctx = mem.get_context()
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import config
from log import logger


# ---------------------------------------------------------------------------
# Padrões de detecção imediata (regex)
# ---------------------------------------------------------------------------

PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # ── Nome ──
    (re.compile(r'(?:meu\s*nome\s*[ée]\s*(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Nome do usuário: {match}'),
    (re.compile(r'(?:eu\s*me\s*chamo\s*(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Nome do usuário: {match}'),
    (re.compile(r'(?:pode\s*me\s*chamar\s*de\s*(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Nome do usuário: {match}'),
    (re.compile(r'(?:me\s*chamo\s*(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Nome do usuário: {match}'),

    # ── Idade ──
    (re.compile(r'(?:tenho\s*|idade\s*de\s*|com\s*)(\d+)\s*anos', re.I),
     'user', 'Idade do usuário: {match} anos'),

    # ── Gostos / Preferências ──
    (re.compile(r'(?:gost[oa]\s*(?:de|muito|mais)?\s*(.+?))(?:[.!?;]|$)', re.I),
     'memory', 'Usuário gosta de {match}'),
    (re.compile(r'(?:ador[oa]\s*(.+?))(?:[.!?;]|$)', re.I),
     'memory', 'Usuário adora {match}'),
    (re.compile(r'(?:prefir[eo]\s*(.+?))(?:[.!?;]|$)', re.I),
     'memory', 'Usuário prefere {match}'),
    (re.compile(r'(?:não\s*gost[oa]\s*(?:de)?\s*(.+?))(?:[.!?;]|$)', re.I),
     'memory', 'Usuário não gosta de {match}'),
    (re.compile(r'(?:odet[eo]\s*(.+?))(?:[.!?;]|$)', re.I),
     'memory', 'Usuário odeia {match}'),
    (re.compile(r'(?:curt[io]\s*(?:de|muito)?\s*(.+?))(?:[.!?;]|$)', re.I),
     'memory', 'Usuário curte {match}'),

    # ── Profissão / Estado civil ──
    (re.compile(r'(?:sou\s+(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Usuário é {match}'),
    (re.compile(r'(?:trabalh[oa]\s*(?:como|em|na|no|com)?\s*(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Usuário trabalha com {match}'),
    (re.compile(r'(?:estudo\s*(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Usuário estuda {match}'),

    # ── Localização ──
    (re.compile(r'(?:moro\s*(?:em|no|na)?\s*(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Usuário mora em {match}'),
    (re.compile(r'(?:sou\s*de\s*(.+?))(?:[.!?;]|$)', re.I),
     'user', 'Usuário é de {match}'),

    # ── Tecnologia / Dispositivos ──
    (re.compile(r'(?:us[oa]\s*(.+?))(?:[.!?;]|$)', re.I),
     'memory', 'Usuário usa {match}'),
    (re.compile(r'(?:tenho\s+um\s+(.+?))(?:[.!?;]|$)', re.I),
     'memory', 'Usuário tem um {match}'),

    # ── Hobbies ──
    (re.compile(r'\btoco\s*(.+?)(?:[.!?;]|$)', re.I),
     'memory', 'Usuário toca {match}'),
    (re.compile(r'\bjog[oa]\b\s*(?:de)?\s*(.+?)(?:[.!?;]|$)', re.I),
     'memory', 'Usuário joga {match}'),
    (re.compile(r'\bassist[eo]\s*(.+?)(?:[.!?;]|$)', re.I),
     'memory', 'Usuário assiste {match}'),
    (re.compile(r'\bleio\b.*?(.+?)(?:[.!?;]|$)', re.I),
     'memory', 'Usuário lê {match}'),

    # ── Sentimentos ──
    (re.compile(r'\bach[oa]\s*(.+?)(?:[.!?;]|$)', re.I),
     'memory', 'Usuário acha {match}'),
    (re.compile(r'\bpens[oa]\s*(.+?)(?:[.!?;]|$)', re.I),
     'memory', 'Usuário pensa {match}'),
]


# ---------------------------------------------------------------------------
# Seção § (delimitador de entradas, igual ao Hermes)
# ---------------------------------------------------------------------------

_SECTION = '\n§\n'


# ---------------------------------------------------------------------------
# Gerenciador
# ---------------------------------------------------------------------------

class MemoryManager:
    """Gerencia memórias em dois arquivos markdown (padrão Hermes).

    Args:
        base_dir: Diretório onde os arquivos são salvos.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir: str = os.path.abspath(base_dir)
        self.memory_path: str = os.path.join(self.base_dir, 'assistant_memory.md')
        self.user_path: str = os.path.join(self.base_dir, 'assistant_user.md')

        self._memory_entries: list[str] = self._load_entries(self.memory_path)
        self._user_entries: list[str] = self._load_entries(self.user_path)

    # ------------------------------------------------------------------
    # Arquivos — entradas separadas por §
    # ------------------------------------------------------------------

    @staticmethod
    def _load_entries(path: str) -> list[str]:
        """Carrega entradas do arquivo (separadas por § ou linhas com -).

        Compatível com os formatos novo (§) e antigo (-).
        """
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Tentar separar por § primeiro
            if '§' in content:
                entries = []
                for part in content.split('§'):
                    part = part.strip()
                    # Ignorar cabeçalhos e linhas vazias
                    if part and not part.startswith('# ') and not part.startswith('#'):
                        entries.append(part)
                return entries

            # Fallback: formato antigo (linhas com - )
            entries = []
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped.startswith('- '):
                    entries.append(stripped[2:].strip())
            return entries

        except IOError:
            return []

    def _save_file(self, path: str, entries: list[str], header_block: str) -> None:
        """Salva entradas com formato Hermes: cabeçalho + entradas § separadas.

        Args:
            path: Caminho do arquivo.
            entries: Lista de entradas de texto.
            header_block: Cabeçalho do arquivo (pode ter múltiplas linhas).
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        try:
            # Aplicar limite de caracteres do config.py
            max_chars = self._get_char_limit(path)
            entries = self._trim_to_limit(entries, max_chars)

            # Montar conteúdo: cabeçalho + entradas separadas por §
            parts = [header_block, '']
            if entries:
                parts.append(_SECTION.join(entries))
                parts.append('')

            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(parts))
        except IOError as e:
            logger.warning(f'Erro ao salvar {path}: {e}')

    def _get_char_limit(self, path: str) -> int:
        """Retorna o limite de caracteres para o arquivo."""
        if path == self.user_path:
            return config.MEMORY_USER_CHAR_LIMIT
        return config.MEMORY_CHAR_LIMIT

    @staticmethod
    def _trim_to_limit(entries: list[str], max_chars: int) -> list[str]:
        """Remove entradas mais antigas até caber no limite."""
        total = sum(len(e) for e in entries)
        while entries and total > max_chars:
            removed = entries.pop(0)
            total -= len(removed)
        return entries

    def _total_chars(self, path: str) -> int:
        """Soma de caracteres de todas as entradas de um arquivo."""
        entries = self._memory_entries if path == self.memory_path else self._user_entries
        return sum(len(e) for e in entries)

    def _usage_pct(self, path: str) -> tuple[int, int, float]:
        """(usados, limite, porcentagem)."""
        used = self._total_chars(path)
        limit = self._get_char_limit(path)
        pct = (used / limit * 100) if limit > 0 else 0
        return (used, limit, pct)

    def save_all(self) -> None:
        """Salva ambos os arquivos."""
        self._save_file(
            self.memory_path, self._memory_entries,
            '# 🧠 Memórias da Chica\n\nAnotações e observações sobre conversas com o usuário.'
        )
        self._save_file(
            self.user_path, self._user_entries,
            '# 👤 Perfil do Usuário\n\nInformações conhecidas sobre o usuário.'
        )

    # ------------------------------------------------------------------
    # CRUD (igual ao Hermes: add, remove)
    # ------------------------------------------------------------------

    def add(self, target: str, fact: str) -> bool:
        """Adiciona um fato à memória.

        Args:
            target: 'user' ou 'memory'.
            fact: Texto do fato.

        Returns:
            True se adicionou, False se já existia.
        """
        fact = fact.strip()
        if not fact or len(fact) > 200:
            return False

        entries = self._user_entries if target == 'user' else self._memory_entries
        path = self.user_path if target == 'user' else self.memory_path

        # Evitar duplicatas (case-insensitive)
        fact_lower = fact.lower()
        for existing in entries:
            if existing.lower() == fact_lower:
                return False

        # Verificar limite antes de adicionar
        if self._total_chars(path) + len(fact) > self._get_char_limit(path):
            # Remover a entrada mais antiga pra abrir espaço
            if entries:
                removed = entries.pop(0)
                logger.info(f'🗑️ Memória removida por limite: {removed[:60]}...')

        entries.append(fact)
        self.save_all()
        logger.info(f'🧠 Memória salva: {fact[:80]}...')
        return True

    def remove(self, target: str, text: str) -> bool:
        """Remove uma entrada por substring.

        Args:
            target: 'user' ou 'memory'.
            text: Substring para buscar.

        Returns:
            True se removeu.
        """
        entries = self._user_entries if target == 'user' else self._memory_entries
        before = len(entries)
        text_lower = text.lower()
        entries[:] = [e for e in entries if text_lower not in e.lower()]
        if len(entries) < before:
            self.save_all()
            return True
        return False

    def clear(self, target: Optional[str] = None) -> None:
        """Limpa memórias.

        Args:
            target: 'user', 'memory' ou None (limpa ambos).
        """
        if target in (None, 'memory'):
            self._memory_entries = []
        if target in (None, 'user'):
            self._user_entries = []
        self.save_all()
        logger.info('🧹 Memórias limpas')

    # ------------------------------------------------------------------
    # Detecção imediata (regex)
    # ------------------------------------------------------------------

    def extract_immediate(self, user_text: str) -> bool:
        """Detecta padrões no texto do usuário e salva imediatamente.

        Returns:
            True se alguma memória foi extraída.
        """
        text_lower = user_text.strip()
        found = False

        for pattern, target, template in PATTERNS:
            m = pattern.search(text_lower)
            if m:
                match_text = m.group(1).strip().title()
                if match_text.isdigit():
                    if len(match_text) < 1 or len(match_text) > 3:
                        continue
                elif len(match_text) < 3 or len(match_text) > 60:
                    continue
                fact = template.format(match=match_text)
                if self.add(target, fact):
                    found = True

        return found

    # ------------------------------------------------------------------
    # Extração via LLM
    # ------------------------------------------------------------------

    def get_extraction_prompt(self) -> str:
        """Retorna o prompt para extrair memórias."""
        return (
            'Analise a conversa abaixo e extraia FATOS IMPORTANTES sobre o usuário '
            'que a assistente Chica deve lembrar para conversas futuras.\n\n'
            'REGRAS:\n'
            '- Retorne APENAS fatos, UM por linha, sem marcadores\n'
            '- Separe em duas seções:\n'
            '  PERFIL: nome, idade, profissão, onde mora, hobbies\n'
            '  MEMORIA: preferências, coisas ditas, opiniões\n'
            '- Se NADA for relevante, retorne apenas "NONE"\n'
            '- NÃO invente — extraia SÓ o que foi dito'
        )

    def apply_extraction(self, extraction_text: str) -> None:
        """Aplica resultado da extração LLM."""
        if not extraction_text or extraction_text.strip().upper() == 'NONE':
            return

        section = None
        added = 0

        for line in extraction_text.split('\n'):
            line = line.strip()
            if line.upper().startswith('PERFIL'):
                section = 'user'
                continue
            elif line.upper().startswith('MEMORIA'):
                section = 'memory'
                continue

            if not line or line.startswith('#'):
                continue

            # Remover marcadores como - ou *
            fact = line.lstrip('- *').strip()
            if not fact or len(fact) > 200:
                continue

            if section and self.add(section, fact):
                added += 1

        if added > 0:
            logger.info(f'🧠 {added} nova(s) memória(s) via LLM')

    # ------------------------------------------------------------------
    # Contexto para o system prompt (estilo Hermes)
    # ------------------------------------------------------------------

    def get_context(self) -> str:
        """Retorna as memórias formatadas como snapshot congelado.

        Formato Hermes:
          ═══════════════════
          MEMORY [67% — chars]
          ═══════════════════
          entrada 1
          §
          entrada 2

        Retorna string vazia se não houver nada.
        """
        parts = []

        if self._memory_entries:
            used, limit, pct = self._usage_pct(self.memory_path)
            header = f'MEMÓRIAS [% — {used}/{limit} chars]'
            sep = '═' * max(len(header), 40)
            block = [
                sep,
                header,
                sep,
                _SECTION.join(self._memory_entries),
            ]
            parts.append('\n'.join(block))

        if self._user_entries:
            used, limit, pct = self._usage_pct(self.user_path)
            header = f'PERFIL DO USUÁRIO [% — {used}/{limit} chars]'
            sep = '═' * max(len(header), 40)
            block = [
                sep,
                header,
                sep,
                _SECTION.join(self._user_entries),
            ]
            parts.append('\n'.join(block))

        if not parts:
            return ''

        return '\n\n📝 INFORMAÇÕES SOBRE O USUÁRIO:\n' + '\n\n'.join(parts)

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self._memory_entries) + len(self._user_entries)

    def filepaths(self) -> dict[str, str]:
        return {
            'memory': self.memory_path,
            'user': self.user_path,
        }
