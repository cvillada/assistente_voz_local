#!/usr/bin/env python3
"""
Executor de comandos locais do sistema operacional com confirmação.

Suporta macOS, Linux e Windows.

Uso:
    from commands import CommandExecutor

    executor = CommandExecutor()
    cmd = executor.parse("abra o navegador")
    if cmd:
        print(f"Comando: {cmd['comando']} -> {cmd['executar']}")
"""

from __future__ import annotations

import platform
import re
import shlex
import subprocess
import sys
from typing import Optional

from log import logger


# ---------------------------------------------------------------------------
# Mapeamento: frase do usuário -> comando do SO
# ---------------------------------------------------------------------------

_COMMANDS: dict[str, dict[str, str]] = {
    # Navegador
    "navegador": {"mac": "open -a Safari", "linux": "xdg-open https://google.com", "win": "start chrome"},
    "browser": {"mac": "open -a Safari", "linux": "xdg-open https://google.com", "win": "start chrome"},
    "chrome": {"mac": "open -a 'Google Chrome'", "linux": "google-chrome", "win": "start chrome"},
    "firefox": {"mac": "open -a Firefox", "linux": "firefox", "win": "start firefox"},
    "safari": {"mac": "open -a Safari", "linux": "", "win": ""},
    "edge": {"mac": "open -a 'Microsoft Edge'", "linux": "microsoft-edge", "win": "start msedge"},
    "brave": {"mac": "open -a Brave Browser", "linux": "brave-browser", "win": "start brave"},

    # Terminal
    "terminal": {"mac": "open -a Terminal", "linux": "x-terminal-emulator", "win": "start cmd"},
    "prompt": {"mac": "open -a Terminal", "linux": "x-terminal-emulator", "win": "start cmd"},

    # Editor de código
    "vscode": {"mac": "code", "linux": "code", "win": "code"},
    "vs code": {"mac": "code", "linux": "code", "win": "code"},
    "visual studio": {"mac": "code", "linux": "code", "win": "code"},
    "cursor": {"mac": "open -a Cursor", "linux": "cursor", "win": "cursor"},
    "sublime": {"mac": "subl", "linux": "subl", "win": "subl"},
    "vim": {"mac": "open -a Terminal vim", "linux": "vim", "win": ""},

    # Música / Mídia
    "spotify": {"mac": "open -a Spotify", "linux": "spotify", "win": "start spotify"},
    "music": {"mac": "open -a Music", "linux": "", "win": ""},
    "vlc": {"mac": "open -a VLC", "linux": "vlc", "win": "start vlc"},

    # Sistema
    "finder": {"mac": "open -a Finder", "linux": "nautilus", "win": "explorer"},
    "explorer": {"mac": "open -a Finder", "linux": "nautilus", "win": "explorer"},
    "arquivos": {"mac": "open -a Finder", "linux": "nautilus", "win": "explorer"},
    "calculadora": {"mac": "open -a Calculator", "linux": "gnome-calculator", "win": "calc"},
    "calendário": {"mac": "open -a Calendar", "linux": "gnome-calendar", "win": "start outlookcal:"},
    "relógio": {"mac": "open -a 'Clock'", "linux": "gnome-clocks", "win": "start ms-clock:"},

    # Redes
    "discord": {"mac": "open -a Discord", "linux": "discord", "win": "start discord"},
    "slack": {"mac": "open -a Slack", "linux": "slack", "win": "start slack"},
    "telegram": {"mac": "open -a Telegram", "linux": "telegram-desktop", "win": "start telegram"},
    "whatsapp": {"mac": "open -a WhatsApp", "linux": "", "win": "start whatsapp:"},

    # Produtividade
    "notion": {"mac": "open -a Notion", "linux": "", "win": "start notion"},
    "obsidian": {"mac": "open -a Obsidian", "linux": "obsidian", "win": "start obsidian"},
    "notas": {"mac": "open -a Notes", "linux": "", "win": ""},
    "lembretes": {"mac": "open -a Reminders", "linux": "", "win": ""},
}


class CommandExecutor:
    """Analisa e executa comandos do sistema com confirmação."""

    def __init__(self) -> None:
        self.system = platform.system().lower()
        if self.system == "darwin":
            self.os_key = "mac"
        elif self.system == "linux":
            self.os_key = "linux"
        elif self.system == "windows":
            self.os_key = "win"
        else:
            self.os_key = "linux"  # fallback

    # ------------------------------------------------------------------
    # Análise do texto
    # ------------------------------------------------------------------

    def parse(self, text: str) -> Optional[dict]:
        """
        Analisa o texto do usuário e retorna um comando se reconhecido.

        Retorna dict com:
            chave:    termo reconhecido (ex: 'navegador')
            comando:  descrição (ex: 'abrir navegador Safari')
            executar: string do comando do SO (ex: 'open -a Safari')
            confirmacao: texto para pedir confirmação (ex: 'Confirma a abertura do navegador?')
        Ou None se não reconheceu nenhum comando.
        """
        text_lower = text.lower().strip()

        # Padrões de abertura: "abra/abre/abrir/open [algo]"
        open_patterns = [
            r"(?:abra|abre|abrir|open|execute|executar|inicia|iniciar|lance|lançar)\s+(?:o\s+|a\s+|)([\w\s]+)",
            r"(?:quero\s+)?(?:abrir|abre|abrir|executar|iniciar|lancar)\s+(?:o\s+|a\s+|)([\w\s]+)",
        ]

        target = None
        for pattern in open_patterns:
            m = re.search(pattern, text_lower)
            if m:
                target = m.group(1).strip()
                break

        # Se não achou com padrão de abertura, tenta match direto
        if not target:
            for key in _COMMANDS:
                if key in text_lower:
                    target = key
                    break

        if not target:
            return None

        # Encontrar qual comando mapeia
        matched_key = None
        for key in _COMMANDS:
            if key in target or target in key:
                matched_key = key
                break

        if not matched_key:
            return None

        cmd_info = _COMMANDS[matched_key]
        cmd_str = cmd_info.get(self.os_key, "")

        if not cmd_str:
            logger.warning(f"Comando '{matched_key}' não suportado em {self.system}")
            return None

        return {
            "chave": matched_key,
            "comando": f"abrir {matched_key}",
            "executar": cmd_str,
            "confirmacao": f"Confirma a abertura do {matched_key}?",
        }

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------

    def execute(self, cmd_info: dict) -> str:
        """
        Executa o comando e retorna uma mensagem de resultado.
        """
        cmd = cmd_info["executar"]
        try:
            if self.os_key == "win":
                subprocess.Popen(cmd, shell=True)
            else:
                subprocess.Popen(shlex.split(cmd))
            logger.success(f"Comando executado: {cmd}")
            return f"Comando executado: {cmd_info['comando']}"
        except Exception as e:
            logger.error(f"Erro ao executar '{cmd}': {e}")
            return f"Não foi possível executar: {e}"
