import pandas as pd
from src.maze.logic import generate_maze, move_player, MazeDecisionTaker


def generate_full_sequence(maze: list[list[int]], maze_id: int, max_steps: int = 500, initial_pos: tuple[int, int] = None) -> list[tuple[str, int, int]]:
    """Given a maze generates a full sequence following a strategy (MazeDecisionTaker)"""
    sequence = []
    player_pos = initial_pos if initial_pos else (0, 0)
    decision_maker = MazeDecisionTaker()

    for i in range(max_steps):
        action = decision_maker(maze, player_pos)
        new_pos = move_player(maze, player_pos, action)
        sequence.append({
                "maze_id": maze_id,
                "step": i,
                "player_pos": player_pos,
                "action": action,
                "new_pos": new_pos
            })
        player_pos = new_pos

    return sequence


def generate_mazes_sequences(num_mazes: int, width: int, max_steps: int = 500, wall_prob: float = 0.2, drop_duplicates: bool = False):
    maze_data = []
    seq_data = []

    for maze_id in range(num_mazes):
        maze = generate_maze(width, (0, 0), wall_prob)
        maze_data.append({"maze_id": maze_id, "maze": maze})
        sequence = generate_full_sequence(maze, maze_id, max_steps)
        seq_data.extend(sequence)

    maze_df, seq_df = pd.DataFrame(maze_data), pd.DataFrame(seq_data)
    if drop_duplicates:
        maze_df = maze_df.drop_duplicates(subset=["maze"], ignore_index=True)
        maze_ids = set(maze_df["maze_id"])
        
        seq_df = seq_df.loc[seq_df["maze_id"].isin(maze_ids)]
        seq_df = seq_df.drop(columns=["step"]).drop_duplicates()

    return maze_df, seq_df
