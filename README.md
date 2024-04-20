


Object Detection with TensorFlow Lite

This repository contains code for performing real-time object detection using TensorFlow Lite. The project includes scripts for  TensorFlow Lite models.

Features:

Real-time object detection using a pre-trained TensorFlow Lite model

Webcam integration for live object detection

Supports various object classes with customizable confidence thresholds


## Object Detection with TensorFlow Lite

This project enables real-time object detection using TensorFlow Lite. Follow the steps below to set up the environment, clone the repository, and run the object detection script.

This can classify objects in 4 classes given in labels.txt

---

### Steps: Set up Conda Environment, Clone the Repository, Install Requirements, and Run the Object Detection Scripts

```bash
conda create --name tflite-env python=3.8 && conda activate tflite-env

git clone https://github.com/AnantKamat12/Custom_object_detection.git

pip install -r requirements.txt

python object_detection.py
#or run command 
python image.py --image <path to image file>

#or run command
python video.py --video <path_to_your_video_file.mp4 >



It is advised to create a new anaconda environment
Clone the repository to your local machine.
Install the required dependencies listed in requirements.txt.
Run the object detection script (object_detection.py) to perform real-time object detection using your webcam.
Feel free to contribute or raise issues if you encounter any problems!


