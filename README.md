# Face Recognition
<i> Work in progress</i>
## Quick start
### Step 1: Clone repository or download zip
`!git clone https://github.com/SarahHannes/face-recognition.git`

### Step 2: Add training images to `\database` folder
### Step 3: Change directory to cloned folder on anaconda prompt
`cd <path to cloned folder>`
### Step 4: Predict! 😌
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
