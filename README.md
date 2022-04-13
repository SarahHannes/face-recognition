# Face Recognition

One shot learning using FaceNet and MTCNN.

## Quick start
### Step 1: Clone repository or <a href="https://github.com/SarahHannes/face-recognition/archive/refs/heads/main.zip">download zip</a>
`!git clone https://github.com/SarahHannes/face-recognition.git`

### Step 2: Add training images to `/database` folder
### Step 3: Create a new environment using <a href="requirements.txt">`requirements.txt`</a> or <a href="face_recog38.yml">`face_recog38.yml`</a>
### Step 4: Activate the created environment on anaconda prompt
`conda activate <environment name>`
### Step 5: Change directory to cloned folder on anaconda prompt
`cd <path to cloned folder>`
### Step 6: Predict! 😌
- To view arguments:
<br>`python main.py --help`
- To set path for training folder (required):
<br>`python main.py --database <path to folder containing training images>`
- To perform face verification through webcam:
<br>`python main.py --database <path to folder containing training images> --webcam`
- To perform face verification on media inputs (.JPG, .MP4):
<br>`python main.py --database <path to folder containing training images> --media <path to folder containing media inputs>`
- To specify threshold:
<br>`python main.py --database <path to folder containing training images> -t <value>`

## References & credits
<i>Thank you!</i>
<br>[1] <a href="https://github.com/foo290/Face-verification-using-One-shot-learning">GitHub repo by foo290</a>
<br>[2] <a href="https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X">GitHub repo by R4j4n</a>
<br>[3] <a href="https://drive.google.com/drive/folders/1-Frhel960FIv9jyEWd_lwY5bVYipizIT">Shared google drive containing FaceNet weights by Hiroki Taniai</a>
<br>[4] <a href="https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/">Tutorial on face recognition</a>
<br>[5] <a href="https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/">Tutorial on argparse</a>
