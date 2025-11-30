import os
from datetime import datetime

def create_run_dir(base_path="/home/peucelle/tpiv-simulations/results"):
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{base_path}/run_{now_str}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_job_id():
    return os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))