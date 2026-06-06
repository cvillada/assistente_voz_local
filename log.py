#!/usr/bin/env python3
"""
Logging colorido e estruturado para o assistente Chica.

Uso:
    from log import logger
    logger.info("Mensagem normal")
    logger.success("Operação concluída")
    logger.warning("Aviso")
    logger.error("Erro")
    logger.header("SEÇÃO")
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from colorama import Fore, Style, init as colorama_init

# Garantir que colorama foi inicializado
colorama_init(autoreset=True)


# ---------------------------------------------------------------------------
# Cores por nível
# ---------------------------------------------------------------------------

LEVEL_COLORS = {
    'DEBUG': Fore.CYAN,
    'INFO': Fore.BLUE,
    'SUCCESS': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'CRITICAL': Fore.RED + Style.BRIGHT,
}

# Nível customizado SUCCESS (entre INFO e WARNING)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')

def success(self, message, *args, **kwargs):
    """Log 'SUCCESS' (nível 25)."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

logging.Logger.success = success  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Formatador colorido
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    """Formata logs com cores baseadas no nível."""

    def format(self, record: logging.LogRecord) -> str:
        cor = LEVEL_COLORS.get(record.levelname, '')
        reset = Style.RESET_ALL

        # Formatar timestamp
        timestamp = self.formatTime(record, self.datefmt or '%H:%M:%S')

        # Símbolo por nível
        simbolos = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '💥',
        }
        simbolo = simbolos.get(record.levelname, '•')

        # Montar linha
        return (
            f"{Fore.CYAN}{timestamp}{reset} "
            f"{cor}{simbolo} {record.getMessage()}{reset}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_logger(name: str = 'chica') -> logging.Logger:
    """Retorna um logger configurado com saída colorida no terminal."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Evitar propagação para o root logger (evita duplicatas)
    logger.propagate = False

    # Handler do console já existente?
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------------
# Logger global padrão
# ---------------------------------------------------------------------------

logger: logging.Logger = get_logger('chica')
