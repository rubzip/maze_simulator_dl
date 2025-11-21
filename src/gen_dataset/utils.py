from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def rotate_action(action: str, rotation: int) -> str:
    action_set = ["up", "right", "down", "left"]
    idx = action_set.index(action)
    new_idx = (idx + rotation) % 4
    return action_set[new_idx]


def rotate_pos(pos: tuple[int, int], rotation: int, width: int) -> tuple[int, int]:
    x, y = pos
    for _ in range(rotation % 4):
        x, y = y, width - 1 - x
    return (x, y)


def rotate_maze(maze: torch.Tensor, rotation: int) -> torch.Tensor:
    return torch.rot90(maze, k=rotation, dims=(0, 1))


def augment_sequences(sequences: pd.DataFrame, width: int) -> pd.DataFrame:
    augmented_sequences = pd.DataFrame(columns=sequences.columns)

    for rotation in range(4):
        aux = sequences.copy()
        aux["maze_id"] = aux["maze_id"].astype(str) + f"_r{rotation}"
        aux["player_pos"] = aux["player_pos"].apply(lambda pos: rotate_pos(pos, rotation, width))
        aux["new_pos"] = aux["new_pos"].apply(lambda pos: rotate_pos(pos, rotation, width))
        aux["action"] = aux["action"].apply(lambda a: rotate_action(a, rotation))
        augmented_sequences = pd.concat([augmented_sequences, aux], ignore_index=True)

    action_to_idx = {
        "up": 0, 
        "right": 1,
        "down": 2, 
        "left": 3
    }

    augmented_sequences["action"] = augmented_sequences["action"].map(action_to_idx)
    return augmented_sequences


def augment_mazes(mazes: pd.DataFrame) -> pd.DataFrame:
    augmented_mazes = pd.DataFrame(columns=mazes.columns)

    for rotation in range(4):
        aux = mazes.copy()
        aux["maze_id"] = aux["maze_id"].astype(str) + f"_r{rotation}"
        aux["maze"] = aux["maze"].apply(lambda m: rotate_maze(m, rotation))
        augmented_mazes = pd.concat([augmented_mazes, aux], ignore_index=True)
    return augmented_mazes


def balance_dataset(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    min_count = df[group_cols].value_counts().min()
    df_balanced = (
        df
        .groupby(group_cols, group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )
    return df_balanced


def build_state(maze: torch.Tensor, pos: tuple[int, int]) -> torch.Tensor:
    """Given a maze (0 void and 1 wall) puts the player (2)"""
    maze_out = maze.clone().to(torch.short)
    x, y = pos
    maze_out[y, x] = 2
    return maze_out


def build_df(mazes_df: pd.DataFrame, sequences_df: pd.DataFrame) -> pd.DataFrame:
    data = []

    for _, row in sequences_df.iterrows():
        maze_tensor = mazes_df.loc[row["maze_id"], "maze"]

        prev_state = build_state(maze_tensor, row["player_pos"])
        actual_state = build_state(maze_tensor, row["new_pos"])
        
        data.append({
            "prev_state": prev_state,
            "action": row["action"],
            "actual_state": actual_state
        })
    return pd.DataFrame(data)


def load_data(path_mazes: str, path_sequences: str):
    mazes_df = pd.read_pickle(Path(path_mazes))
    sequences_df = pd.read_pickle(Path(path_sequences))
    return mazes_df, sequences_df


def split_data(mazes_df):
    train_id, test_id = train_test_split(
        mazes_df["maze_id"].unique(), test_size=0.2, random_state=42
    )

    return train_id, test_id


def process_data(mazes_df, sequences_df, width: int):
    group_cols = ["player_pos", "action"]
    
    mazes_df["maze"] = mazes_df["maze"].apply(lambda m: torch.tensor(m, dtype=torch.long))
    mazes_df = augment_mazes(mazes_df)

    sequences_df = augment_sequences(sequences_df, width)
    sequences_df = balance_dataset(sequences_df, group_cols)
    sequences_df = build_df(mazes_df.set_index("maze_id"), sequences_df)

    return mazes_df, sequences_df
