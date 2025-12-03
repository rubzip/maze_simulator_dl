import numpy as np
import pygame
import torch
import torch.nn.functional as F
from src.gen_dataset.utils import ACTION_TO_IDX
from src.maze.logic import generate_maze, move_player
from settings import CELL_SIZE, BACKGROUND_COLOR, WALL_COLOR, PLAYER_COLOR


class GameNN:
    def __init__(self, model: torch.nn.Module, width: int = 20, wall_prob: float = 0.2):
        self.width = width
        self.model = model
        self.wall_prob = wall_prob

        pygame.init()
        self.screen = pygame.display.set_mode((self.width * CELL_SIZE, self.width * CELL_SIZE))
        pygame.display.set_caption("Maze Simmulation")

        self.clock = pygame.time.Clock()
        self.running = True

        self.reset()
    
    def reset(self):
        self.state = np.array(generate_maze(self.width, (0, 0), wall_prob=self.wall_prob))
        self.state[0, 0] = 2

    def run(self, fps: int = 10):
        while self.running:
            self.clock.tick(fps)
            self.check_reset()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            action = self.get_action()
            action_idx = ACTION_TO_IDX.get(action)
            if action_idx is not None:
                self.next_state(action_idx)

            self.screen.fill(BACKGROUND_COLOR)
            for y in range(self.width):
                for x in range(self.width):
                    if self.state[y][x] == 1:
                        self.draw_rect(x, y, WALL_COLOR)
                    if self.state[y][x] == 2:
                        self.draw_rect(x, y, PLAYER_COLOR)

            pygame.display.flip()
        pygame.quit()
    
    def next_state(self, action_idx: int):
        state_tensor = torch.tensor(self.state).long()
        state_one_hot = F.one_hot(state_tensor, num_classes=3).float().permute(2, 0, 1).unsqueeze(0)
        action_tensor = torch.tensor(action_idx).long()
        action_one_hot = F.one_hot(action_tensor, num_classes=4).float().unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            new_state_logits = self.model(state_one_hot, action_one_hot)
        self.state = torch.argmax(new_state_logits, dim=1).squeeze(0).cpu().numpy()
    
    def draw_rect(self, x: int, y: int, color: tuple[int, int, int]):
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect)

    def check_reset(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            self.__init__(self.model, self.width)
        if keys[pygame.K_q]:
            self.running = False

    def get_action(*args) -> str:
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

from src.models.unet import GameUNet
from settings import WIDTH, WALL_PROB, FPS

if __name__ == "__main__":
    state_dict = torch.load("model/model.pth")
    model = GameUNet()
    model.load_state_dict(state_dict)
    game = GameNN(model=model, width=WIDTH, wall_prob=WALL_PROB)
    game.run(fps=FPS)
