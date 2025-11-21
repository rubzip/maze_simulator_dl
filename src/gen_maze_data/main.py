from pathlib import Path
import pandas as pd
from .utils import generate_mazes_sequences
from settings import WIDTH, NUM_MAZES, MAX_STEPS, WALL_PROB, DATA_PATH, DROP_DUPLICATES


if __name__ == "__main__":
    mazes_df, seq_df = generate_mazes_sequences(NUM_MAZES, WIDTH, MAX_STEPS, WALL_PROB, DROP_DUPLICATES)

    folder_path = Path(DATA_PATH)
    folder_path.mkdir(parents=True, exist_ok=True)

    mazes_df.to_pickle(folder_path / "mazes.pkl")
    seq_df.to_pickle(folder_path / "sequences.pkl")

    print("Datasets saved")
