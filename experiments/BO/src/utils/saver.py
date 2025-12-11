import os
import pickle
import pandas as pd
import logging

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def update_summary_csv(run_dir, run_idx):
    summary_csv = os.path.join(run_dir, "summary.csv")
    logs_csv = os.path.join(run_dir, f"logs_{run_idx}.csv")

    if not os.path.isfile(logs_csv):
        logging.warning("logs.csv not found")
        return

    try:
        df = pd.read_csv(logs_csv)
        df.to_csv(summary_csv,
                  mode='a' if os.path.isfile(summary_csv) else 'w',
                  header=not os.path.isfile(summary_csv),
                  index=False)
    except Exception as e:
        logging.warning(f"Failed to update summary CSV: {e}")