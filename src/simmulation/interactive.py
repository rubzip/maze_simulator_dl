import pygame
from .utils import Game
from settings import WIDTH, WALL_PROB, FPS


def controller_strategy(*args) -> str:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return "up"
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return "down"
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return "left"
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return "right"
    return "None"


if __name__ == "__main__":
    game = Game(strategy=controller_strategy, width=WIDTH, wall_prob=WALL_PROB)
    game.run(fps=FPS)
