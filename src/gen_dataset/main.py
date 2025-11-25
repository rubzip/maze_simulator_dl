from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from settings import DATA_PATH, DATASET_PATH
from .utils import augment_sequences, augment_mazes, balance_dataset, build_dataset


if __name__ == "__main__":
    mazes_df = pd.read_pickle(Path(DATA_PATH) / 'mazes.pkl')
    mazes_df["maze"] = mazes_df["maze"].apply(lambda m: torch.tensor(m, dtype=torch.long))

    sequences_df = pd.read_pickle(Path(DATA_PATH) / 'sequences.pkl')
    train_ids, test_ids = train_test_split(
        mazes_df["maze_id"].unique(), test_size=0.2, random_state=42
    )

    train_mazes = mazes_df[mazes_df["maze_id"].isin(train_ids)]
    test_mazes = mazes_df[mazes_df["maze_id"].isin(test_ids)]

    train_val_sequences = sequences_df[sequences_df["maze_id"].isin(train_ids)]
    test_sequences = sequences_df[sequences_df["maze_id"].isin(test_ids)]

    width = len(mazes_df["maze"][0])
    height = len(mazes_df["maze"][0][0])

    del mazes_df, sequences_df

    group_cols = ["player_pos", "action"]

    train_val_sequences_augmented = augment_sequences(train_val_sequences, width)
    train_mazes_augmented = augment_mazes(train_mazes)
    train_val_sequences_balanced = balance_dataset(train_val_sequences_augmented, group_cols)

    test_sequences_augmented = augment_sequences(test_sequences, width)
    test_mazes_augmented = augment_mazes(test_mazes)
    test_sequences_balanced = balance_dataset(test_sequences_augmented, group_cols)

    del train_mazes, test_mazes
    del train_val_sequences, test_sequences
    del train_val_sequences_augmented, test_sequences_augmented

    train_val_sequences_balanced["stratify_key"] = (
        train_val_sequences_balanced[group_cols]
        .astype(str)
        .agg("_".join, axis=1)
    )

    train_sequences_balanced, val_sequences_balanced = train_test_split(
        train_val_sequences_balanced,
        test_size=0.2,
        random_state=42,
        stratify=train_val_sequences_balanced["stratify_key"]
    )

    del train_val_sequences_balanced

    Path(DATASET_PATH).mkdir(parents=True, exist_ok=True)

    train = build_dataset(train_mazes_augmented.set_index("maze_id"), train_sequences_balanced)
    torch.save(train, Path(DATASET_PATH) / "train.pt")
    del train
    
    test = build_dataset(test_mazes_augmented.set_index("maze_id"), test_sequences_balanced)
    torch.save(test, Path(DATASET_PATH) / "test.pt")
    del test

    val = build_dataset(train_mazes_augmented.set_index("maze_id"), val_sequences_balanced)
    torch.save(val, Path(DATASET_PATH) / "val.pt")
    del val
