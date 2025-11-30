import logging, os, sys
from datetime import datetime

def setup_logger(run_dir, run_index, verbose=True):
    """
    Sets up a logger that logs to both a file and the console.
    Args:
        run_dir (str): Directory where log files will be saved.
        run_index (int): Index of the current run/experiment.
        verbose (bool): If True, sets logger to DEBUG level; otherwise INFO level.
    Returns:
        logger (logging.Logger): Configured logger instance.
        log_file (str): Path to the log file.
    """
    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_file = os.path.join(run_dir, f"experiment_{run_index}_{job_id}.log")
    logger = logging.getLogger(f"logger_{run_index}_{job_id}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)

    return logger, log_file