from .utils import Game
from src.maze.logic import MazeDecisionTaker
from settings import WIDTH, WALL_PROB, FPS


if __name__ == "__main__":
    selector = MazeDecisionTaker()
    game = Game(strategy=selector, width=WIDTH, wall_prob=WALL_PROB)
    game.run(fps=FPS)
