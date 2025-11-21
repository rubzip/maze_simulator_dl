import pygame
from src.maze.logic import generate_maze, move_player
from settings import CELL_SIZE, BACKGROUND_COLOR, WALL_COLOR, PLAYER_COLOR


class Game:
    def __init__(self, strategy: callable, width: int = 20, wall_prob: float = 0.2):
        self.width = width
        self.strategy = strategy
        self.wall_prob = wall_prob

        pygame.init()
        self.screen = pygame.display.set_mode((self.width * CELL_SIZE, self.width * CELL_SIZE))
        pygame.display.set_caption("Maze Simmulation")

        self.clock = pygame.time.Clock()
        self.running = True

        self.reset()
    
    def reset(self):
        self.maze = generate_maze(self.width, (0, 0), wall_prob=self.wall_prob)
        self.pos = (0, 0)

    def run(self, fps: int = 10):
        while self.running:
            self.clock.tick(fps)

            self.check_reset()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            action = self.strategy(self.maze, self.pos)
            self.pos = move_player(self.maze, self.pos, action)

            # Draw Background
            self.screen.fill(BACKGROUND_COLOR)
            for y in range(self.width):
                for x in range(self.width):
                    if self.maze[y][x] == 1:
                        # Draw Walls
                        self.draw_rect(x, y, WALL_COLOR)

            # Draw Player
            self.draw_rect(self.pos[0], self.pos[1], PLAYER_COLOR)
            pygame.display.flip()
        pygame.quit()
    
    def draw_rect(self, x: int, y: int, color: tuple[int, int, int]):
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect)

    def check_reset(self):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.__init__(self.strategy, self.width)
            if keys[pygame.K_q]:
                self.running = False

