import subprocess, threading
import os, time, random
import pandas as pd
import uuid
import socket

# Constants
NUM_NODES = 1
NUM_GPUS = 4
MAX_RETRIES = 1
TIME_TO_HANG = 1000
error_dict = {
    'CUDA_OOM': "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate",
    "DeepSpeed_OOM": "RuntimeError: CUDA error: out of memory",
    'IndexError': "IndexError: list index out of range", 
    "TimeoutError": "TimeoutError: The client socket has timed out after "
}


# Load job configurations from CSV
def get_size_time(log_file_path):
    last_check_time = time.time()
    if os.path.exists(log_file_path):
        last_size = os.path.getsize(log_file_path)
    else:
        last_size = 0
    return last_check_time, last_size

def check_errors_in_file(file_path):
    """
    Checks for multiple error strings in a text file.

    Args:
    file_path (str): The path to the text file to be checked.
    error_dict (dict): A dictionary where keys are error names and values are the strings to search for.

    Returns:
    dict: A dictionary with error names as keys and boolean values indicating whether the error string was found.
    """
    if not os.path.exists(file_path):
        return False, None

    results = {error: False for error in error_dict}
    with open(file_path, 'r') as file:
        for line in file:
            for error_name, search_string in error_dict.items():
                if search_string in line:
                    return True, error_name

    return False, None


def find_free_port() -> int:
    """
    Find and return an available port on the local machine.

    This function creates a temporary socket, binds it to a free port assigned
    by the operating system, and then closes the socket. The assigned port 
    number is returned.

    Returns:
        int: A free port number available for use.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# Function to submit and monitor a single job
def submit_and_monitor_job(row, df):

    os.makedirs('./slurm_logs', exist_ok=True)

    # RDZV_ID = str(random.randint(0, 9999)) # trunkate to 4 digits
    # RDZV_PORT= random.randint(20000, 29999)  # Range from 20000 to 29999
    RDZV_ID = str(uuid.uuid4())[:8]  # Truncate for readability
    RDZV_PORT = find_free_port()  # Get an available port

    BATCH_SCRIPT = f"torchrun{RDZV_PORT}.sh"
    print("writing", BATCH_SCRIPT)

    with open(BATCH_SCRIPT, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=dir
#SBATCH -p pasteur
#SBATCH -A pasteur
#SBATCH --nodes={NUM_NODES}
#SBATCH --ntasks={NUM_NODES}
#SBATCH --mem=64GB
#SBATCH --gres=gpu:{NUM_GPUS} # gpu:a100{NUM_GPUS}
#SBATCH --cpus-per-task={2 * NUM_GPUS}
#SBATCH --output=./slurm_logs/overfit-%j-out.txt
#SBATCH --error=./slurm_logs/overfit-%j-err.txt
#SBATCH --exclude=pasteur[1-4,8-9]

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export OMP_NUM_THREADS=1
export FI_PROVIDER=efa

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_GROUP="ViT_L_14_from_webli"

source ~/.bashrc
conda activate train_clip3

cd ../src

srun torchrun --nproc_per_node={NUM_GPUS} --nnodes={NUM_NODES} --node_rank=$SLURM_NODEID --rdzv_id {RDZV_ID} --rdzv_backend c10d --rdzv_endpoint $head_node_ip:{RDZV_PORT} -m open_clip_train.main \
    --save-most-recent \
    --train-data "/pasteur2/u/ale9806/data/pmc-oa/full_panel_w_labels/commercial/{{000000..000100}}.tar::/pasteur2/u/ale9806/data/pmc-oa/full_panel_w_labels/noncommercial/{{000000..000100}}.tar::/pasteur2/u/ale9806/data/pmc-oa/full_panel_w_labels/other/{{000000..000100}}.tar" \
    --train-num-samples 30000 \
    --accum-freq 4 \
    --lr-scheduler "cosine" \
    --dataset-type "webdataset" \
    --lr 1e-6 \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 1000 \
    --wd 0.2 \
    --batch-size 512 \
    --epochs 100 \
    --workers 1 \
    --model "ViT-L-14" \
    --precision "fp16" \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 1 \
    --seed 0 \
    --logs ./logs_exp_dir/ \
    --pretrained "commonpool_xl_clip_s13b_b90k" \
    --report-to "wandb" \
    --wandb-project-name "open-biomed-clip"
""")

    # Submit the job
    sbatch_output = subprocess.check_output(["sbatch", BATCH_SCRIPT]).decode("utf-8")
    job_id = sbatch_output.split()[-1].strip()
    log_file_path = f"./logs/apollo_{job_id}.out" 
    print(f"Job {job_id} submitted.")

    # Update the CSV with the job ID
    df.loc[df.index == row.name, 'jobID'] = job_id
    df.loc[df.index == row.name, 'status'] = 0 # 0 for running
    df.to_csv('jobs_post.csv', index=False)
    last_check_time, last_size = get_size_time(log_file_path)
    # Monitor the job status
    RETRY_COUNT = 0
    while RETRY_COUNT <= MAX_RETRIES:
        time.sleep(30)
        cur_check_time, cur_size = get_size_time(log_file_path)
        if last_size==cur_size:
            if cur_check_time - last_check_time > TIME_TO_HANG and cur_size > 0:
                os.system(f"scancel {job_id}")
                print(f"Job {job_id} is hanging!!!")
                break
        else:
            last_check_time, last_size = cur_check_time, cur_size

        job_state = subprocess.check_output(["squeue", "--job", job_id, "--noheader", "--format=%T"]).decode("utf-8").strip()
        print(f"Job {job_id} state is '{job_state}'")
        status, error = check_errors_in_file(log_file_path)

        if job_state == "COMPLETED":
            df.loc[df.index == row.name, 'status'] = "COMPLETED"  # 1 for completed
            print(f"Job {job_id} completed successfully.")
            break
        elif job_state in ["PENDING"]:
            #print(f"Job {job_id} is pending.")
            df.loc[df.index == row.name, 'status'] = job_state  # -1 for not running
            continue

        elif job_state in ["FAILED", "CANCELLED", "TIMEOUT"]:
            df.loc[df.index == row.name, 'status'] = job_state  # -1 for not running
            os.system(f"scancel {job_id}")
            print(f"Job {job_id} ended with state {job_state}.")
            RETRY_COUNT += 1
            break
        elif status:
            os.system(f"scancel {job_id}")
            print(f"Job {job_id} ended with state {error}.")
            RETRY_COUNT += 1
            break
        else:
            # Check if the job is hanging
            print("unknown state", job_state)
    # Update the CSV after job completion or failure
    df.to_csv('jobs_post.csv', index=False)

    # Optionally, remove the temporary batch script
    os.remove(BATCH_SCRIPT)


if __name__=='__main__':
    df = pd.read_csv('experiments.csv')
    print("found", len(df), "jobs")
    threads = []
    for index, row in df.iterrows():
        thread = threading.Thread(target=submit_and_monitor_job, args=(row, df,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All jobs have been processed.")


