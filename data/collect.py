import os
import cv2
import numpy as np
import face_recognition
from bing_image_downloader import downloader

### All data will be stored in this folder ###
model_dnn = '../dnn_model.caffemodel'
prototxt = '../deploy.prototxt'
DATA_DIR = './dataset/bing'
TEST_DATA_DIR = './dataset/test'

### the models ###
detector = cv2.dnn.readNetFromCaffe(prototxt, model_dnn)

search_queries = {
	'Kobe Bryan' : 'Male',
	'Jacksepticeye' : 'Male',
	'Donald Trump' : 'Male',
	'Hillary Clinton' : 'Female',
	'Barack Obama' : 'Male',
	'Jack Ma' : 'Male',
	'Bill Gates' : 'Male',
	'Elon Musk' : 'Male',
	'Emma Watson' : 'Female',
	'Lee Hsien Loong' : 'Male',
	'Marina Joyce' : 'Female',
    'Kanye West' : 'Male',
    'Taylor Swift' : 'Female',
    'Jeff Bezoos' : 'Male',
    'Selena Gomez' : 'Female',
    'Justin Beiber' : 'Male',
    'Ariana Grande' : 'Female',
    'Andrew Ng' : 'Male',
    'Markiplier' : 'Male',
    'Billie Eilish' : 'Female',
    'Beyonce' : 'Female',
    'Jackie Chan' : 'Male',
    'Stephen Chow' : 'Male',
    'Fan BingBing' : 'Female',
    'Yang Mi' : 'Female'
}


print("[INFO] Downloading images ... ")
names = list(search_queries.keys())
for query in names:
	if(os.path.exists(DATA_DIR + "/" + query)):
		print("[INFO] Data dir '" + query + "' has already exist, Skipping ... ")
		continue
		
	downloader.download(query, limit=35, adult_filter_off=False, force_replace=True)

### After the download we need to check for validity ###
### Criteria : only 1 face per image, true gender ### 

print("[INFO] Checking data validity ... ")
ALL_SATISFIED = False
while(not ALL_SATISFIED):
	invalid_count = 0
	for (dir, dirs, files) in os.walk(DATA_DIR):
		if(dir != DATA_DIR):
			for file in files:
				abs_path = dir + "/" + file
				print("[INFO] Check image file " + abs_path)

				img = cv2.imread(abs_path)

				if(isinstance(img, type(None))):
					if(os.path.exists(abs_path)):
						os.remove(abs_path)
					continue 

				(H, W) = img.shape[:2]

				blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), (104,111,123))
				detector.setInput(blob)
				detections = detector.forward()

				num_faces = 0
				for i in range(0, detections.shape[2]):
					confidence = detections[0,0,i,2]
					if(confidence > 0.5):
						num_faces += 1

				if(num_faces != 1):
					print("[INFO] Invalid image at : " + abs_path + ", removing image ... ")
					invalid_count += 1
					os.remove(abs_path)

	if(invalid_count == 0):
		ALL_SATISFIED = True

### move some of the data to to testing folder ###
print("[INFO] Creating testing dataset ... ")
if(not os.path.exists(TEST_DATA_DIR)):
	print("[INFO] Testing data directory does not exist, creating ... ")
	os.mkdir(TEST_DATA_DIR)

for (dir, dirs, files) in os.walk(DATA_DIR):
	# each class will have 30% of testing image w.r.t original dataset
	index = np.random.choice(len(files), size=int(len(files)/3), replace=False)
	test_files = np.array(files)[index]

	for file in test_files:
		old_abs_path = dir + "/" + file
		class_folder = dir.split("/")[-1]

		if(not os.path.exists(TEST_DATA_DIR + "/" + class_folder)):
			os.mkdir(TEST_DATA_DIR + "/" + class_folder)

		new_abs_path = TEST_DATA_DIR + "/" + class_folder + "/" + file 

		if(os.path.exists(old_abs_path)):
			print("[INFO] Moving " + old_abs_path + " to " + new_abs_path)
			os.rename(old_abs_path, new_abs_path)