# SCITAS EPFL Cluster User Guide

This guide provides a **complete and practical overview** of using the SCITAS cluster at EPFL. It covers everything from initial setup and file transfers to job submission, monitoring, retrieving results, managing Python environments, and useful tips for efficient cluster usage.

---

## 1️⃣ Setup: Connecting to the SCITAS Cluster

Before using the cluster, you need to connect and set up secure access.

### Why connect?

Connecting to the cluster allows you to run computations on powerful remote resources.

### Steps to connect

- **Open a terminal** on your Mac (use Terminal app or any emulator).
- **Connect via SSH** by running:

  ```bash
  ssh your_username@izar.hpc.epfl.ch
  ```

  Replace `your_username` with your actual EPFL username. You may be prompted for your password or SSH key passphrase.

- **Optional: Set up SSH keys for passwordless login** to avoid entering your password every time:

  1. Generate an SSH key pair (if you don't have one):

     ```bash
     ssh-keygen -t ed25519 -C "your_email@example.com"
     ```

  2. Copy your public key to the cluster:

     ```bash
     ssh-copy-id your_username@izar.hpc.epfl.ch
     ```

---

## 2️⃣ File Transfer: Moving Files Between Your Mac and the Cluster

Efficient file transfer is essential to upload your data and retrieve results.

### Why transfer files?

You need to send scripts, data, and resources to the cluster and download output files after computations.

### Methods to transfer files

#### Using `scp` (secure copy)

- Send a file from your Mac to the cluster:

  ```bash
  scp myfile.txt your_username@izar.hpc.epfl.ch:~
  ```

- Send a directory recursively:

  ```bash
  scp -r myfolder your_username@izar.hpc.epfl.ch:~
  ```

- Retrieve a file from the cluster to your Mac:

  ```bash
  scp your_username@izar.hpc.epfl.ch:output.txt .
  ```

- Retrieve a directory recursively:

  ```bash
  scp -r your_username@izar.hpc.epfl.ch:myfolder .
  ```

#### Using `rsync` (efficient synchronization)

`rsync` transfers only changed parts of files, making repeated syncs faster.

- Send files/directories from Mac to cluster:

  ```bash
  rsync -avz --progress myfolder/ your_username@izar.hpc.epfl.ch:~/myfolder/
  ```

- Retrieve files/directories from cluster to Mac:

  ```bash
  rsync -avz --progress your_username@izar.hpc.epfl.ch:~/myfolder/ myfolder/
  ```

- Useful options explained:

  - `-a`: archive mode (preserves permissions, symbolic links, etc.)
  - `-v`: verbose output
  - `-z`: compress data during transfer
  - `--progress`: show progress during transfer

---

## 3️⃣ Job Submission: Running Your Tasks on the Cluster

The SCITAS cluster uses the **SLURM** workload manager to schedule and manage jobs efficiently.

### Why submit jobs?

You submit jobs to request computing resources and run your scripts on the cluster.

### How to submit jobs

#### Step 1: Create a SLURM batch script (e.g., `job.sh`)

Write a script that specifies resource requests and commands to run.

##### Example: CPU job

```bash
#!/bin/bash
#SBATCH --job-name=my_cpu_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=01:00:00          # Max runtime (HH:MM:SS)
#SBATCH --partition=standard     # Partition name
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=4        # Number of CPU cores per task

module load python/3.8           # Load Python module (adjust version if needed)

python myscript.py
```

##### Example: GPU job

```bash
#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --output=output_gpu.txt
#SBATCH --error=error_gpu.txt
#SBATCH --time=02:00:00
#SBATCH --partition=gpu           # GPU partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # Request 1 GPU

module load cuda/11.2            # Load CUDA module if needed
module load python/3.8

python my_gpu_script.py
```

#### Step 2: Submit the job

Run the following command to submit your job script:

```bash
sbatch job.sh
```

You will receive a job ID in response.

---

## 4️⃣ Monitoring Jobs: Tracking Your Job Status

Monitoring helps you know if your jobs are running, pending, or completed.

### How to monitor jobs

- **List your jobs:**

  ```bash
  squeue -u your_username
  ```

  Shows all jobs submitted by you, including JobID, Partition, Name, State, Time, Nodes, etc.

- **Get detailed information about a specific job:**

  ```bash
  scontrol show job <job_id>
  ```

  Replace `<job_id>` with your actual job number.

- **Check job history and resource usage:**

  ```bash
  sacct -j <job_id>
  ```

### Common job states

- `COMPLETED` ✅ — Job finished successfully
- `FAILED` ❌ — Job ended with an error
- `CANCELLED` 🟡 — Job was cancelled by user or admin
- `TIMEOUT` ⏱️ — Job exceeded its time limit
- `PENDING` — Waiting in queue
- `RUNNING` — Currently executing

---

## 5️⃣ Retrieving Results: Downloading Output Files

Once your job finishes, you need to retrieve results for analysis.

### How to get your results

- Copy output files from the cluster to your Mac:

  ```bash
  scp your_username@izar.hpc.epfl.ch:output.txt .
  ```

- Download entire output directories:

  ```bash
  scp -r your_username@izar.hpc.epfl.ch:results_folder .
  ```

---

## 6️⃣ Python Environments: Managing Your Software Setup

Using Python virtual environments helps you manage packages without affecting the system.

### Why manage Python environments?

It ensures reproducibility and avoids conflicts between package versions.

### Using environment modules

- List available modules:

  ```bash
  module avail
  ```

- Load a module (e.g., Python 3.8):

  ```bash
  module load python/3.8
  ```

- Check loaded modules:

  ```bash
  module list
  ```

- Unload a module:

  ```bash
  module unload python/3.8
  ```

### Creating and using Python virtual environments

1. Load the Python module:

   ```bash
   module load python/3.8
   ```

2. Create a virtual environment (replace `myenv` with your desired name):

   ```bash
   python -m venv myenv
   ```

3. Activate the virtual environment:

   ```bash
   source myenv/bin/activate
   ```

4. Install packages with pip:

   ```bash
   pip install numpy scipy matplotlib
   ```

5. Deactivate when done:

   ```bash
   deactivate
   ```

6. In your SLURM job script, activate the environment before running your Python script:

   ```bash
   source ~/myenv/bin/activate
   python myscript.py
   ```

---

## 7️⃣ Tips and Best Practices for Using the Cluster

### Erasing files safely

To remove files or directories on the cluster:

```bash
rm -rf ~/myfolder
```

- `rm` = remove
- `-r` = recursive (delete all files and subdirectories)
- `-f` = force (no confirmation prompts)

**⚠️ Be very careful with this command!** Double-check the path before running to avoid deleting important data.

### Additional useful commands

- Check disk usage:

  ```bash
  du -sh ~/*
  ```

- Check available storage quota:

  ```bash
  quota -s
  ```

- Use `screen` or `tmux` for persistent sessions:

  ```bash
  screen -S mysession
  ```

  or

  ```bash
  tmux new -s mysession
  ```

- Cancel a running job:

  ```bash
  scancel <job_id>
  ```

---

## 8️⃣ Example SLURM Script Explained

Here is an example SLURM batch script:

```bash
#!/bin/bash
#SBATCH --job-name=BO_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/peucelle/tpiv-simulations/results/run_%j/log_%j_%a.txt
#SBATCH --error=/home/peucelle/tpiv-simulations/results/run_%j/err_%j_%a.txt
#SBATCH --chdir /home/peucelle/tpiv-simulations/experiments
#SBATCH --array=0-9
#SBATCH --mem-per-cpu=9000
```

### Explanation of each row

| Row                                   | Directive                                     | Description                                                                                         |
|-------------------------------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `#!/bin/bash`                       | Shebang                                      | Specifies that the script should be run using Bash.                                                |
| `#SBATCH --job-name=BO_exp`         | Job name                                     | Sets a name for your job (`BO_exp`), useful for monitoring.                                        |
| `#SBATCH --nodes=1`                 | Number of nodes                              | Requests 1 compute node.                                                                            |
| `#SBATCH --ntasks=1`                | Number of tasks                             | Number of parallel tasks (processes) to run.                                                      |
| `#SBATCH --cpus-per-task=1`         | CPUs per task                               | Number of CPU cores assigned to each task.                                                        |
| `#SBATCH --gres=gpu:1`              | GPU resources                              | Requests 1 GPU for the job.                                                                        |
| `#SBATCH --time=24:00:00`           | Maximum runtime                            | Limits the job to 24 hours (HH:MM:SS).                                                           |
| `#SBATCH --output=/home/peucelle/tpiv-simulations/results/run_%j/log_%j_%a.txt` | Standard output file                         | File where stdout is written. `%j` is the job ID, `%a` is the array index.                         |
| `#SBATCH --error=/home/peucelle/tpiv-simulations/results/run_%j/err_%j_%a.txt`    | Standard error file                          | File where stderr is written. `%j` is the job ID, `%a` is the array index.                         |
| `#SBATCH --chdir /home/peucelle/tpiv-simulations/experiments`                     | Working directory                           | Directory where the script will run.                                                              |
| `#SBATCH --array=0-9`              | Job array                                   | Submits 10 jobs (indexes 0 to 9) in an array, useful for running multiple experiments.            |
| `#SBATCH --mem-per-cpu=9000`        | Memory per CPU                             | Allocates 9000 MB (9 GB) of RAM per CPU core.                                                     |

---

If you have any questions or encounter issues, please contact the HPC support team at EPFL.

Happy computing! 🚀