import logging
import os

def init_logging(run_dir, verbose=True):
    log_file = os.path.join(run_dir, "experiment.txt")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

def log_header(run_idx, job_id, device, run_dir, config):
    header = [
        "========================================",
        "        EXPERIMENT START",
        "----------------------------------------",
        f"Run index: {run_idx}",
        f"Job ID: {job_id}",
        f"Device: {device}",
        f"Results directory: {run_dir}",
        "========================================",
    ]
    msg = "\n".join(header)
    print(msg)
    logging.info(msg)

def log_footer(run_dir, run_idx):
    footer = [
        "----------------------------------------",
        "Experiment finished.",
        f"Results saved in {run_dir}",
        f"Logs CSV: logs_{run_idx}.csv",
        "----------------------------------------",
    ]
    msg = "\n".join(footer)
    print(msg)
    logging.info(msg)