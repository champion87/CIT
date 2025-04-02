from datetime import datetime
import os
import subprocess
from time import sleep
import time
from run_experiment import run_experiment
from PIL import Image
import glob

test_experiments = ["bed", "face", "cactus", "dog", "tiger"]
more_experiments = ["fish1", "fish2", "guitar", "dragon"]

def run_all_experiments(experiments):
	for name in experiments:
		run_experiment(name)
    
def wait_for_jobs():
	while True:
		result = subprocess.run(["squeue", "--me"], capture_output=True, text=True)
		if "killable" not in result.stdout:
			break
		time.sleep(10)
    
    

def get_results_from_tmp():		
	# Get all PNG images in /tmp
	image_paths = sorted(glob.glob("./tmp/*.png"))

	# Open images and concatenate them vertically
	images = [Image.open(img_path) for img_path in image_paths]

	if len(images) == 0:
		print("No images were found in /tmp. Exiting.")
		exit(0)

	total_height = sum(img.height for img in images)
	max_width = max(img.width for img in images)

	concatenated_image = Image.new("RGB", (max_width, total_height))
	y_offset = 0
	for img in images:
		concatenated_image.paste(img, (0, y_offset))
		y_offset += img.height

	# Save the concatenated image
	concatenated_image.save(f"./results/system_test_at_{datetime.now().strftime('%d.%b.%Y-%H:%M:%S')}.png")

	# Delete the original images
	for img_path in image_paths:
		os.remove(img_path)
	
if __name__ == "__main__":
	# run_all_experiments(test_experiments)
	run_all_experiments(more_experiments)
	print("\n\nAll jobs have been submitted. Waiting for all jobs to finish...\n")
	wait_for_jobs()
	get_results_from_tmp()
	print("All jobs have finished. The results have been concatenated and saved as a PNG image.")