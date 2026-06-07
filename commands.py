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

        # Se não achou com padrão de abertura, tenta match direto,
        # mas com heurísticas para evitar falsos positivos
        # (ex: "calendário dos jogos" não deve abrir o Calendar)
        match_type = "explicit"  # veio dos open_patterns
        if not target:
            match_type = "implicit"  # veio do fallback
            for key in _COMMANDS:
                if key in text_lower:
                    # Heurística: palavra-chave seguida de preposição "de/do/da/dos/das"
                    # indica "algo DE algo" (ex: "calendário dos jogos") → pergunta, não comando
                    key_pattern = re.compile(
                        re.escape(key) + r'\s+(d[eoas]s?)\b',
                        re.IGNORECASE
                    )
                    if key_pattern.search(text_lower):
                        continue  # provável pedido de informação, não comando

                    # Heurística: frase contém palavras interrogativas → é pergunta
                    question_words = [
                        r'\b(que|qual|quais|como|quando|onde|porque|por\s*que|quem)\b',
                        r'\b(pode|poderia)\s+me\s+(dizer|mostrar|falar|informar)\b',
                        r'\b(quero\s+saber|gostaria\s+de\s+saber|me\s+diga)\b',
                        # "você pode X" + "?" → pedido/pergunta
                        r'\b(você|vc)\s+(pode|poderia)\s+\w+\s*$',
                        # "com usar/fazer/encontrar..." → pedido de instrução
                        r'\b(como|onde)\s+(usar|fazer|encontrar|instalar|configurar)\b',
                    ]
                    is_question = any(
                        re.search(qw, text_lower) for qw in question_words
                    )

                    # Heurística: ponto de interrogação no final → é pergunta
                    if not is_question and text_lower.rstrip().endswith('?'):
                        is_question = True

                    # Heurística: frase contém verbos de pesquisa/pedido de informação
                    search_verbs = [
                        r'\b(pesquis[aeiou]|pesquisa[rz]?)\w*\b',
                        r'\b(procur[aeiou]|procura[rz])\w*\b',
                        r'\b(busc[aeiou]|busca[rz])\w*\b',
                    ]
                    has_search_verb = any(
                        re.search(sv, text_lower) for sv in search_verbs
                    )

                    # Heurística: frase longa (> 5 palavras) SEM verbo de ação
                    # indica conversa genérica, não comando
                    word_count = len(text_lower.split())
                    has_action_verb = any(
                        re.search(r'\b' + v + r'\b', text_lower)
                        for v in ['abra', 'abre', 'abrir', 'open', 'execute',
                                  'executar', 'inicia', 'iniciar', 'lance', 'lançar']
                    )
                    is_long_phrase = word_count > 5 and not has_action_verb

                    # Heurística: verbo "mostrar/mostre/mostra" ou "ler/ver/olhar"
                    # antes ou depois da keyword → pedido de informação
                    context_verbs = [
                        r'\b(mostr[aeio]|mostre|exib[ei])\b',
                        r'\b([lt]er|olhar|conferir)\b',
                    ]
                    has_context_verb = any(
                        re.search(cv, text_lower) for cv in context_verbs
                    )

                    # Só aceita o match se NÃO parecer pergunta/informação
                    if not is_question and not has_search_verb and not is_long_phrase \
                            and not has_context_verb:
                        target = key
                        break

        if not target:
            return None

        # Heurística extra para matches EXPLÍCITOS (open_patterns):
        # Se a palavra capturada está ao lado de "pesquisa/pesquise/sobre",
        # ou a frase contém palavras interrogativas, provavelmente não é
        # um comando puro de abrir app.
        # Ex: "abra uma pesquisa no navegador sobre python" → não deve abrir navegador
        # Ex: "como abrir o terminal?" → é pergunta, não comando
        if match_type == "explicit":
            # Verificar se há palavras interrogativas na frase
            explicit_question = re.search(
                r'\b(que|qual|quais|como|quando|onde|porque|por\s*que|quem)\b',
                text_lower
            )
            if explicit_question:
                logger.info(f'🧠 Match explícito ignorado (contém pergunta): {text_lower[:60]}')
                return None

            # Verificar se há indicadores de pesquisa perto do target
            research_indicators = [
                r'\b(pesquis[aeiou]|pesquisar?)\w*\b',
                r'\b(sobre|a\s*respeito\s*de|acerca\s*de)\b',
                r'\b(procura[rz]|busca[rz])\w*\b',
            ]
            has_research = any(
                re.search(ri, text_lower) for ri in research_indicators
            )
            if has_research:
                # Verificar se a palavra de pesquisa está perto do target
                target_idx = text_lower.find(target)
                target_end = target_idx + len(target)
                nearby = text_lower[max(0, target_idx-30):target_end+30]
                if any(re.search(ri, nearby) for ri in research_indicators):
                    logger.info(f'🧠 Match explícito ignorado (contém pesquisa): {text_lower[:60]}')
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
            "match_type": match_type,  # "explicit" ou "implicit"
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
