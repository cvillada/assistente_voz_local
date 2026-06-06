#!/usr/bin/env python3
"""
Módulo do avatar animado da assistente Chica.

Gerencia a janela Pygame com animações de piscar e falar,
sincronizadas com o estado da assistente.
"""

from __future__ import annotations

import math
import os
import platform
import threading
import time

import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, QUIT

import config
from colorama import Fore


class AvatarManager:
    """Gerencia a janela do avatar com animações."""

    def __init__(self, image_dir: str | None = None) -> None:
        self.image_dir = image_dir if image_dir else config.AVATAR_IMAGE_DIR
        self.images: dict[str, pygame.Surface] = {}
        self.current_state: str = "normal"
        self.window: pygame.Surface | None = None
        self.screen: pygame.Surface | None = None
        self.running: bool = False
        self.animation_thread: threading.Thread | None = None
        self.is_speaking: bool = False
        self.blink_timer: float = 0
        self.blink_interval: float = config.AVATAR_BLINK_INTERVAL
        self.last_blink_time: float = time.time()
        self.speak_animation_timer: float = 0
        self.speak_animation_speed: float = config.AVATAR_SPEAK_ANIMATION_SPEED

        # Detectar SO
        self.is_macos: bool = platform.system() == 'Darwin'

        # Inicializar Pygame no macOS imediatamente (thread principal)
        if self.is_macos:
            try:
                pygame.init()
                pygame.display.init()
                print(Fore.YELLOW + "⚠️  Pygame inicializado para macOS (thread principal)")
            except Exception as e:
                print(Fore.RED + f"❌ Erro ao inicializar Pygame no macOS: {e}")

        self.states: dict[str, str] = config.AVATAR_STATES

    # ------------------------------------------------------------------
    # Imagens
    # ------------------------------------------------------------------

    def load_images(self) -> bool:
        """Carrega todas as imagens do avatar. Retorna True se OK."""
        try:
            for state, filename in self.states.items():
                image_path = os.path.join(self.image_dir, filename)
                if os.path.exists(image_path):
                    self.images[state] = pygame.image.load(image_path)
                    print(Fore.GREEN + f"✅ Imagem carregada: {filename}")
                else:
                    print(Fore.RED + f"❌ Imagem não encontrada: {image_path}")
                    return False
            return True
        except Exception as e:
            print(Fore.RED + f"❌ Erro ao carregar imagens: {e}")
            return False

    # ------------------------------------------------------------------
    # Janela
    # ------------------------------------------------------------------

    def init_window(self) -> bool:
        """Inicializa a janela do avatar. Retorna True se OK."""
        if not config.AVATAR_ENABLE:
            print(Fore.YELLOW + "⚠️  Avatar desabilitado, ignorando init_window()")
            return False

        try:
            width, height = config.get_avatar_window_size()
            if self.is_macos:
                return self._init_window_macos(width, height)
            else:
                pygame.init()
                self.window = pygame.display.set_mode((width, height))
                return self._finish_window_init(width, height)
        except Exception as e:
            print(Fore.RED + f"❌ Erro ao inicializar janela do avatar: {e}")
            return False

    def _init_window_macos(self, width: int, height: int) -> bool:
        """Inicializa a janela no macOS (deve rodar na thread principal)."""
        try:
            print(Fore.YELLOW + "⚠️  macOS: criando janela na thread principal...")
            if threading.current_thread() == threading.main_thread():
                print(Fore.YELLOW + "✅ Já estamos na thread principal")
                self.window = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                pygame.display.set_caption(f"Avatar {config.ASSISTANT_NAME}")
                return self._finish_window_init(width, height)
            else:
                print(Fore.YELLOW + "⚠️  Não estamos na thread principal, adiando criação da janela")
                self.window = None
                self.pending_window_size = (width, height)
                self.pending_window_caption = f"Avatar {config.ASSISTANT_NAME}"
                return True
        except Exception as e:
            print(Fore.RED + f"❌ Erro ao criar janela no macOS: {e}")
            return False

    def _finish_window_init(self, width: int, height: int) -> bool:
        """Finaliza a inicialização da janela (comum a todos os SOs)."""
        try:
            if not self.load_images():
                return False
            for state in self.images:
                self.images[state] = pygame.transform.scale(
                    self.images[state], (width, height)
                )
            self.screen = pygame.display.get_surface()
            self.running = True
            print(Fore.GREEN + f"✅ Janela do avatar inicializada ({width}x{height})")
            return True
        except Exception as e:
            print(Fore.RED + f"❌ Erro ao finalizar inicialização da janela: {e}")
            return False

    # ------------------------------------------------------------------
    # Animação
    # ------------------------------------------------------------------

    def set_speaking(self, speaking: bool) -> None:
        """Define se o avatar está falando."""
        if not config.AVATAR_ENABLE:
            return
        self.is_speaking = speaking
        if speaking:
            self.speak_animation_timer = 0

    def update_animation(self) -> None:
        """Atualiza a animação (piscar e/ou falar)."""
        now = time.time()

        # Piscar (quando não está falando)
        if not self.is_speaking:
            if now - self.last_blink_time > self.blink_interval:
                self.current_state = "olho"
                self.last_blink_time = now
                self.blink_timer = 0.2
            elif self.blink_timer > 0:
                self.blink_timer -= 0.016
                if self.blink_timer <= 0:
                    self.current_state = "normal"

        # Animação de fala
        if self.is_speaking:
            self.speak_animation_timer += self.speak_animation_speed
            if math.sin(self.speak_animation_timer) > 0:
                self.current_state = "boca"
            else:
                self.current_state = "normal"

    # ------------------------------------------------------------------
    # Renderização
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Renderiza o avatar na tela."""
        if not self.running or not self.screen:
            return
        self.screen.fill((255, 255, 255))
        if self.current_state in self.images:
            self.screen.blit(self.images[self.current_state], (0, 0))
        pygame.display.flip()

    def handle_events(self) -> None:
        """Processa eventos da janela."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.running = False

    def update_and_render(self) -> bool:
        """Atualiza e renderiza o avatar. Deve ser chamado periodi­camente."""
        if not config.AVATAR_ENABLE:
            return False

        # Criar janela pendente (macOS)
        if self.is_macos and self.window is None and hasattr(self, 'pending_window_size'):
            try:
                print(Fore.YELLOW + "🪟 Criando janela do avatar na primeira atualização...")
                width, height = self.pending_window_size
                self.window = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                pygame.display.set_caption(self.pending_window_caption)
                if self.load_images():
                    for state in self.images:
                        self.images[state] = pygame.transform.scale(
                            self.images[state], (width, height)
                        )
                    self.screen = pygame.display.get_surface()
                    print(Fore.GREEN + f"✅ Janela do avatar criada ({width}x{height})")
                else:
                    print(Fore.RED + "❌ Falha ao carregar imagens para a janela")
                    return False
                del self.pending_window_size
                del self.pending_window_caption
            except Exception as e:
                print(Fore.RED + f"❌ Erro ao criar janela na atualização: {e}")
                return False

        if not self.running or not self.screen:
            return False

        try:
            self.handle_events()
            self.update_animation()
            self.render()
            return True
        except Exception as e:
            print(Fore.YELLOW + f"⚠️  Erro ao atualizar avatar: {e}")
            return False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia o avatar."""
        if not config.AVATAR_ENABLE:
            print(Fore.YELLOW + "⚠️  Avatar desabilitado, ignorando start()")
            return
        print(Fore.GREEN + f"✅ Avatar {config.ASSISTANT_NAME} inicializado")
        if self.is_macos:
            print(Fore.YELLOW + "⚠️  macOS: Avatar rodando na thread principal")
        else:
            print(Fore.YELLOW + "⚠️  Use update_and_render() periodicamente para animar o avatar")

    def stop(self) -> None:
        """Para o avatar."""
        if not config.AVATAR_ENABLE:
            return
        self.running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1.0)
        pygame.quit()
