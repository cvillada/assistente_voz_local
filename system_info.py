#!/usr/bin/env python3
"""
Informações do sistema para a assistente Chica.

Executa comandos locais para responder perguntas sobre o sistema:
  - Espaço em disco, memória RAM, CPU
  - Sistema operacional, temperatura, bateria
  - Rede, processos, uptime

Usa subprocess (sem dependências externas).
Funciona em macOS, Linux e Raspberry Pi.
"""

from __future__ import annotations

import platform
import re
import subprocess
from typing import Callable, Optional

from log import logger

# ---------------------------------------------------------------------------
# Comandos por sistema operacional
# ---------------------------------------------------------------------------

def _os() -> str:
    return platform.system().lower()

# ---------------------------------------------------------------------------
# Funções de execução
# ---------------------------------------------------------------------------

def _run(cmd: str) -> str:
    """Executa um comando shell e retorna a saída."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        return output if output else result.stderr.strip()
    except subprocess.TimeoutExpired:
        return '(comando excedeu o tempo limite)'
    except Exception as e:
        return f'(erro: {e})'


def _disk_info() -> str:
    """Informação de espaço em disco."""
    cmd = 'df -h /' if _os() != 'windows' else 'wmic logicaldisk get size,freespace,caption'
    return _run(cmd)


def _memory_info() -> str:
    """Informação de memória RAM total e uso."""
    if _os() == 'darwin':
        total = _run("sysctl -n hw.memsize | awk '{printf \"%.1f GB\", $1/1073741824}'")
        used = _run("vm_stat | awk '/Pages active/ {a=$NF} /Pages wired/ {w=$NF} END {printf \"%.1f GB usados\", (a+w)*4096/1073741824}'")
        return f'Total: {total}\nUso: {used}'
    elif _os() == 'linux':
        return _run('free -h')
    return _run('free -h' if _os() != 'windows' else 'systeminfo | findstr Memory')


def _cpu_info() -> str:
    """Informação de CPU."""
    if _os() == 'darwin':
        logical = _run('sysctl -n hw.logicalcpu')
        physical = _run('sysctl -n hw.physicalcpu')
        brand = _run("sysctl -n machdep.cpu.brand_string")
        return f'Processador: {brand}\nNúcleos físicos: {physical}\nNúcleos lógicos: {logical}'
    elif _os() == 'linux':
        return _run('lscpu | grep -E "Model name|CPU(s)|Thread"')
    return _run('wmic cpu get name,numberofcores')


def _os_info() -> str:
    """Informação do sistema operacional."""
    if _os() == 'darwin':
        return _run('sw_vers')
    elif _os() == 'linux':
        return _run('cat /etc/os-release 2>/dev/null || lsb_release -a')
    return f'{platform.system()} {platform.release()} ({platform.version()})'


def _temp_info() -> str:
    """Informação de temperatura/thermal (macOS e Linux)."""
    if _os() == 'darwin':
        return _run('pmset -g therm')
    elif _os() == 'linux':
        return _run('sensors 2>/dev/null || echo "sensors não disponível"')
    return '(não suportado neste sistema)'


def _battery_info() -> str:
    """Informação de bateria."""
    if _os() == 'darwin':
        return _run('pmset -g batt')
    elif _os() == 'linux':
        return _run('acpi -b 2>/dev/null || upower -i $(upower -e | grep BAT) 2>/dev/null || echo "bateria não detectada"')
    return '(não suportado neste sistema)'


def _ip_info() -> str:
    """Informação de rede/IP."""
    if _os() == 'darwin':
        ip = _run("ifconfig | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}'")
        return f'IP local:\n{ip}' if ip else 'Nenhum IP ativo encontrado'
    elif _os() == 'linux':
        return _run("hostname -I 2>/dev/null || ip addr show | grep 'inet ' | grep -v 127.0.0.1")
    return _run('ipconfig | findstr IPv4')


def _uptime_info() -> str:
    """Tempo de atividade do sistema."""
    return _run('uptime')


# ---------------------------------------------------------------------------
# Mapa de consultas: (palavras-chave, função_ou_comando, descrição)
# ---------------------------------------------------------------------------

class SystemQuery:
    """Representa uma consulta de sistema detectada."""

    def __init__(self, name: str, runner: Callable[[], str]) -> None:
        self.name = name
        self._runner = runner

    def execute(self) -> str:
        return self._runner()


_QUERIES: list[tuple[list[str], SystemQuery]] = [
    # Disco
    (['espaço em disco', 'espaço no disco', 'espaço do disco',
      'armazenamento', 'disco livre', 'disco usado', 'quanto espaço',
      'hd cheio', 'disco rigido', 'disco rígido'],
     SystemQuery('espaço em disco', _disk_info)),

    # Memória RAM
    (['memória ram', 'memoria ram', 'quantidade de ram', 'quanta ram',
      'memória total', 'memoria total', 'memória do pc', 'memoria do pc',
      'memoria', 'memória', 'uso de memoria', 'uso de memória',
      'memoria usada', 'memoria disponivel'],
     SystemQuery('memória RAM', _memory_info)),

    # CPU / Processador
    (['cpu', 'processador', 'nucleos', 'núcleos',
      'quantos nucleos', 'quantos núcleos', 'quantos cores',
      'velocidade do processador'],
     SystemQuery('CPU', _cpu_info)),

    # Sistema operacional
    (['sistema operacional', 'versão do sistema', 'qual sistema',
      'qual so', 'qual OS', 'versão do mac', 'versao do mac',
      'qual mac', 'qual linux', 'qual windows'],
     SystemQuery('sistema operacional', _os_info)),

    # Temperatura
    (['temperatura', 'aquecimento', 'esquentando', 'thermal'],
     SystemQuery('temperatura', _temp_info)),

    # Bateria
    (['bateria', 'carga da bateria', 'nível da bateria',
      'nivel da bateria', 'battery', 'quanto de bateria',
      'bateria carregando'],
     SystemQuery('bateria', _battery_info)),

    # Rede / IP
    (['rede', 'ip', 'endereço ip', 'endereco ip',
      'conexão', 'conexao', 'conexão de rede',
      'meu ip', 'qual meu ip'],
     SystemQuery('rede', _ip_info)),

    # Processos
    (['processos', 'programas rodando', 'aplicativos abertos',
      'o que está rodando', 'o que esta rodando', 'programas abertos',
      'quais programas', 'tarefas'],
     SystemQuery('processos', lambda: _run('ps aux --sort=-%mem | head -15'))),

    # Uptime
    (['uptime', 'quanto tempo ligado', 'tempo ligado',
      'desde quando', 'quanto tempo o pc', 'horas ligado'],
     SystemQuery('uptime', _uptime_info)),

    # Usuário
    (['quem sou eu', 'nome do usuário', 'nome do usuario',
      'usuário atual', 'usuario atual', 'quem está logado',
      'qual usuario', 'minha conta'],
     SystemQuery('usuário', lambda: f'Usuário: {_run("whoami")}\nHome: {_run("echo $HOME")}')),

    # Pasta home / diretórios
    (['pasta home', 'diretório home', 'diretorio home', 'minha pasta',
      'meu diretorio', 'meu diretório'],
     SystemQuery('diretório home', lambda: f'Home: {_run("echo $HOME")}')),
]


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------

def detect_system_query(text: str) -> Optional[SystemQuery]:
    """Verifica se o texto do usuário é uma consulta sobre o sistema.

    Args:
        text: Texto do usuário.

    Returns:
        SystemQuery se detectado, None caso contrário.
    """
    text_lower = text.lower().strip()

    for keywords, query in _QUERIES:
        for kw in keywords:
            if kw in text_lower:
                logger.info(f'🖥️ Consulta de sistema detectada: {query.name}')
                return query

    return None


# ---------------------------------------------------------------------------
# Formatação para o prompt
# ---------------------------------------------------------------------------

def format_for_prompt(name: str, output: str) -> str:
    """Formata a saída do comando para incluir no prompt do LLM.

    Args:
        name: Nome da consulta (ex: 'espaço em disco').
        output: Saída do comando executado.

    Returns:
        String formatada para o prompt.
    """
    if not output:
        return ''
    return (
        f'\n\n📊 INFORMAÇÃO DO SISTEMA — {name.upper()}:\n'
        f'{output}\n'
        f'\n(Explique estes dados ao usuário de forma natural em português.'
        f' Seja direto, use o sistema métrico.'
        f' Não mencione que executou um comando a menos que perguntem.)'
    )
