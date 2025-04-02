import glob
import shutil
import os



EXPERIMENTS_FOLDER = "experiments"
LOGS_FOLDER = "logs"

def delete_all_logs():
	os.system(f"rm -rf {LOGS_FOLDER}/*")
 
def delete_all_experiments():
	for path in glob.glob(os.path.join(EXPERIMENTS_FOLDER, "*/MVD*")):
		print(f"Deleting {path}")
		if input(f"Deleting {path}. Are you sure? (y/n) ") == "y":
			shutil.rmtree(path, ignore_errors=True)  # Safer deletion

if __name__ == "__main__":
	delete_all_logs()
	delete_all_experiments()
	print("All experiments and logs have been deleted.")