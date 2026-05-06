from roboflow import Roboflow
import os

# Initialize with your API key
rf = Roboflow(api_key="MozIyzBJ7nMRkP0x2EmF")


# 2. Access the Pothole project
# Based on the best Indian road dataset we found earlier
project = rf.workspace("pothole-detector-dmdod").project("pothole-clzln")
version = project.version(1)

# 3. Download the Dataset
# This will create a folder named 'pothole-dataset' in your current directory
# You can change 'yolov8' to your preferred format (e.g., 'coco', 'voc')
print("Downloading dataset...")
dataset = version.download(model_format="yolov8", location="./pothole-dataset")

# 4. Access the Model
# This prepares the pre-trained model for local inference
model = version.model

print(f"\nSetup Complete!")
print(f"Dataset downloaded to: {os.path.abspath('./pothole-dataset')}")
print("You can now use 'model.predict()' on images or video frames.")

# Quick test on a local image if you have one
# result = model.predict("your_test_image.jpg", confidence=40)
# result.save("test_prediction.jpg")
