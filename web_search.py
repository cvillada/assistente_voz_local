#!/usr/bin/env python3
"""
Busca na web para a assistente Chica.

Usa DuckDuckGo HTML (gratuito, sem API key).
Não depende de pacotes externos — apenas 'requests' (já comum no ambiente).

Uso:
    from web_search import search_web, needs_search

    if needs_search("Qual o clima hoje?"):
        resultados = search_web("clima hoje")
        # resultados → string formatada
"""

from __future__ import annotations

import re
from typing import Optional

from log import logger

# ---------------------------------------------------------------------------
# Palavras-chave que indicam necessidade de pesquisa na web
# ---------------------------------------------------------------------------

SEARCH_KEYWORDS: list[str] = [
    'pesquise', 'pesquisa', 'pesquisar', 'pesquisei',
    'procure', 'procurar', 'busque', 'buscar',
    'internet', 'google', 'na web', 'online',
    'clima', 'tempo', 'previsão', 'previsao',
    'temperatura', 'chover', 'chove',
    'notícias', 'noticias', 'últimas', 'ultimas',
    'novidades', 'lançamento',
    'quem é', 'quem foi', 'o que é', 'o que foi',
    'qual a', 'qual o', 'quais',
    'onde fica', 'como usar', 'como fazer',
    'quando foi', 'quando é',
    'qual o significado', 'qual o preço', 'quanto custa',
]

SEARCH_SUFFIXES: list[str] = [
    'na internet', 'no google', 'no site',
    'da atualidade', 'do momento',
    'última hora', 'ultima hora',
]

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_search_cache: dict[str, str] = {}
_MAX_CACHE = 20


# ---------------------------------------------------------------------------
# Verificação de conectividade
# ---------------------------------------------------------------------------

def check_internet() -> bool:
    """Verifica se há acesso à internet (tentativa rápida a um host confiável).

    Returns:
        True se tiver internet, False caso contrário.
    """
    import socket
    hosts = [
        ('1.1.1.1', 443),       # Cloudflare
        ('8.8.8.8', 443),       # Google DNS
        ('duckduckgo.com', 443), # DuckDuckGo
    ]
    for host, port in hosts:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((host, port))
            s.close()
            return True
        except (socket.timeout, OSError):
            continue
    return False


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------

def needs_search(text: str) -> bool:
    """Verifica se o texto indica necessidade de pesquisa na web."""
    text_lower = text.lower().strip()

    for suffix in SEARCH_SUFFIXES:
        if suffix in text_lower:
            return True

    for kw in SEARCH_KEYWORDS:
        if kw in text_lower:
            return True

    return False


# ---------------------------------------------------------------------------
# Busca principal (requests + DuckDuckGo HTML)
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = 5) -> str:
    """Pesquisa na web usando DuckDuckGo HTML.

    Args:
        query: Termo de pesquisa.
        max_results: Máximo de resultados (padrão: 5).

    Returns:
        String formatada com os resultados, ou vazia se falhar.
    """
    cache_key = query.lower().strip()
    if cache_key in _search_cache:
        logger.info('🔍 Resultado carregado do cache')
        return _search_cache[cache_key]

    try:
        import requests
    except ImportError:
        logger.warning('Pacote requests não disponível')
        return ''

    from urllib.parse import quote_plus

    url = f'https://html.duckduckgo.com/html/?q={quote_plus(query)}'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
    }

    try:
        logger.info(f'🔍 Pesquisando: {query}')
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        # Tentar uma vez mais (rate limit do DuckDuckGo)
        try:
            import time as t
            t.sleep(1)
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e2:
            logger.warning(f'Erro na requisição: {e2}')
            return ''

    html = resp.text

    # Extrair resultados com regex
    # Padrão: link do resultado → título → snippet
    results: list[tuple[str, str, str]] = []

    # Buscar blocos de resultado com regex
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        r'.*?'
        r'(?:class="result__snippet"[^>]*>(.*?)</(?:a|div)>'
        r'|class="result__body"[^>]*>.*?class="result__snippet"[^>]*>(.*?)</)',
        re.DOTALL,
    )

    for m in pattern.finditer(html):
        if len(results) >= max_results:
            break

        href = _clean_url(m.group(1))
        title = _strip_html(m.group(2)).strip()
        # Pega o snippet do grupo 3 ou 4 (qual capturou)
        snippet = _strip_html(m.group(3) or m.group(4) or '').strip()

        if title:
            results.append((title, snippet, href))

    if not results:
        logger.info('🔍 Nenhum resultado encontrado')
        return ''

    # Formatar
    lines: list[str] = []
    for i, (title, snippet, href) in enumerate(results, 1):
        lines.append(f'{i}. {title}')
        if snippet:
            if len(snippet) > 200:
                snippet = snippet[:197] + '...'
            lines.append(f'   {snippet}')
        lines.append('')

    result_text = '\n'.join(lines).strip()

    # Cache
    if len(_search_cache) < _MAX_CACHE:
        _search_cache[cache_key] = result_text

    logger.success(f'🔍 {len(results)} resultado(s) encontrados')
    return result_text


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """Limpa o cache de pesquisas."""
    _search_cache.clear()
    logger.info('🧹 Cache de pesquisa limpo')


def format_for_prompt(results: str) -> str:
    """Formata resultados para inclusão no prompt do LLM."""
    if not results:
        return ''
    return (
        '\n\n📰 RESULTADOS DA PESQUISA NA WEB:\n'
        f'{results}\n'
        '\n(Baseie sua resposta nestas informações. '
        'Use linguagem natural — não mencione que fez pesquisa a menos que perguntem.)'
    )


def _clean_url(url: str) -> str:
    """Limpa uma URL (remove redirecionamento do DuckDuckGo)."""
    # DuckDuckGo usa redirecionamento: //duckduckgo.com/l/?uddg=URL_ENCONDED&...
    m = re.search(r'uddg=([^&]+)', url)
    if m:
        from urllib.parse import unquote
        return unquote(m.group(1))
    return url


def _strip_html(text: str) -> str:
    """Remove tags HTML de um texto."""
    return re.sub(r'<[^>]+>', '', text).strip()
