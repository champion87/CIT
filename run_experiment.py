import sys
import os
from datetime import datetime

def run_slurm(python_line, name):
	script_path = f"experiments/{name}/{name}_CIT.slurm"
	script_header = f"""#! /bin/sh
#SBATCH --job-name={name}_CIT
#SBATCH --output=logs/{name}_at_{datetime.now().strftime('%H:%M:%S')}/{name}_CIT.out
#SBATCH --error=logs/{name}_at_{datetime.now().strftime('%H:%M:%S')}/{name}_CIT.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=24G


"""
	script_content = script_header + python_line

	# Ensure the experiment directory exists
	os.makedirs(f"experiments/{name}", exist_ok=True)

	# Write the script to a file
	with open(script_path, "w") as script_file:
		script_file.write(script_content)

	# Submit the job using sbatch
	os.system(f"sbatch {script_path}")

	# print(f"SLURM script written to {script_path}")
	print (script_content)
	print("\n\nThe experiment has been submitted to the cluster.")

	os.remove(script_path)
	print(f"SLURM script removed from {script_path}")

def notify_empty_queue():
    is_queue_not_empty = os.system("squeue --me | grep -q killable")
    check_loop = f"while {is_queue_not_empty}; do sleep 10; echo 'Job Still Running' done; echo 'Job finished!'"
    os.system(check_loop)

if __name__ == "__main__":
	name = sys.argv[1]
	# python_line = f"python src/cit_run.py --config experiments/{name}/config.yaml"
	python_line = f"python run_smvd.py --config experiments/{name}/config.yaml"
	# python_line = f"python run_smvd.py --config data/{name}/config.yaml"
 
	if len(sys.argv) > 2:
		os.system(python_line)
	else:
		run_slurm(python_line, name)
		notify_empty_queue()
	