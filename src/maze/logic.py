import random
from typing import Literal
from collections import defaultdict


def generate_maze(width: int, init_pos: tuple[int, int] = None, wall_prob: float = 0.2) -> tuple[tuple[int]]:
    """Generates a maze with the given width and wall probability.

    Args:
        width (int): The width of the maze (width x width).
        init_pos (tuple[int, int], optional): The initial position of the player. Defaults to None.
        wall_prob (float, optional): The probability of a cell being a wall. Defaults to 0.2.

    Returns:
        tuple[tuple[int]]: The generated maze. 0 represents free space, 1 represents a wall.
    """
    x0, y0 = init_pos if init_pos else (0, 0)
    maze = tuple([
        tuple([
            0 if random.random() > wall_prob or (x == x0 and y == y0) else 1
            for x in range(width)
        ])
        for y in range(width)
    ])
    return maze


def move_player(maze: tuple[tuple[int]], pos: tuple[int, int], action: Literal["up", "down", "left", "right"]) -> tuple[int, int]:
    """Moves the player in the maze according to the action."""
    x, y = pos
    if action == "up": nx, ny = x, y-1
    elif action == "down": nx, ny = x, y+1
    elif action == "left": nx, ny = x-1, y
    elif action == "right": nx, ny = x+1, y
    else: nx, ny = x, y

    if 0 <= nx < len(maze) and 0 <= ny < len(maze) and maze[ny][nx] == 0:
        return (nx, ny)
    return (x, y)


class MazeDecisionTaker:
    """
    Decides the next move in a maze, prioritizing less-visited positions.
    """
    DIRECTIONS = [(0, -1, "up"), (0, 1, "down"), (-1, 0, "left"), (1, 0, "right")]

    def __init__(self, random_prob: float = 0.2):
        self.visited_counts = defaultdict(int)
        self.random_prob = random_prob
    
    def __call__(self, maze: tuple[tuple[int]], pos: tuple[int, int]) -> Literal["up", "down", "left", "right"]:
        width = len(maze)
        x, y = pos

        self.visited_counts[pos] += 1

        valid_moves = []
        invalid_moves = []
        for dx, dy, action in self.DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < width and maze[ny][nx] == 0:
                valid_moves.append((nx, ny, action))
            else:
                invalid_moves.append((nx, ny, action))

        if (invalid_moves and random.random() < self.random_prob) or not valid_moves:
            return random.choice(invalid_moves)[2]

        min_visits = min(self.visited_counts[(nx, ny)] for nx, ny, _ in valid_moves)
        least_visited_moves = [action for nx, ny, action in valid_moves if self.visited_counts[(nx, ny)] == min_visits]

        return random.choice(least_visited_moves)
