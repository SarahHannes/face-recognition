import os
import time

from mtcnn import MTCNN
import numpy as np
import cv2
import h5py
import tensorflow as tf
from architecture_facenet import InceptionResNetV2

tf.config.experimental.set_visible_devices([], 'GPU')


def get_xy(box):
  """
  Get bounding box result.
  """
  x1, y1, width, height = box
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = (x1 + width), (y1 + height)
  return x1, y1, x2, y2

def get_face(image, resize_scale=(160, 160)):
  """
  Output the face result.
  """
  face_list = []
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # Use MTCNN to detect all faces appeared in an image
  faces_detected = face_detector.detect_faces(image)
  
  # Loop through each individual face detection result
  for detected_face in faces_detected:
    x1, y1, x2, y2 = get_xy(detected_face['box'])
    # Crop out the face
    final_face = image[y1:y2, x1:x2]
    # Resize the image
    face_array = cv2.resize(final_face, resize_scale)
    face_list.append(face_array)
  
  # Return a list of detected face image and output of MTCNN
  return face_list, faces_detected

def get_face_embeddings(face_image):
  """
  Produce embeddings using FaceNet.
  """
  # Change pixels' datatype
  face_image = face_image.astype('float32')
  # Normalize pixel value because the model only takes in normalized values
  mean, std = face_image.mean(), face_image.std()
  face_image = (face_image - mean) / std
  # Expand the first dimension so that the model can take in (model require batch)
  samples = np.expand_dims(face_image, axis=0)
  # Use the facenet to produce embeddings
  embeddings = model.predict(samples)
  return embeddings

def load_saved_user():
  """
  Load images from database.
  """
  # Create empty list to store the embeddings of the face in the database and the name
  saved_faces = []
  saved_faces_name = []
  # List down all the images in the database
  face_database = os.listdir(face_database_path)
  
  # If database is not empty
  # Read the images, then use FaceNet to produce the embeddings for comparison later
  if face_database:
    for face_img in face_database:
      # Load image
      image_np = cv2.imread(os.path.join(face_database_path, face_img))
      face_list, detected_faces = get_face(image_np)
      # Assuming only 1 entry per face in the database
      face_embedding = get_face_embeddings(face_list[0])
      # Save embeddings and names in list
      saved_faces.append(face_embedding)
      saved_faces_name.append(face_img.split('.')[0])
  else:
    print('[WARNING] Database is empty')
  return saved_faces, saved_faces_name

def mark_face(detected_face, image, x1, x2, y1, y2):
  """
  Draw all detected bounding boxes and keypoints on image.
  """
  # Get the face keypoints
  left_eye = detected_face['keypoints']['left_eye']
  right_eye = detected_face['keypoints']['right_eye']
  nose = detected_face['keypoints']['nose']
  mouth_left = detected_face['keypoints']['mouth_left']
  mouth_right = detected_face['keypoints']['mouth_right']

  # Draw bounding box
  image = cv2.rectangle(image, (x1, y1), (x2, y2), color=COLOR, thickness=THICKNESS)
  # Draw facial keypoints. thickness=-1 gives a filled circle
  image = cv2.circle(image, left_eye, radius=2, color=COLOR, thickness=-1)
  image = cv2.circle(image, right_eye, radius=2, color=COLOR, thickness=-1)
  image = cv2.circle(image, nose, radius=2, color=COLOR, thickness=-1)
  image = cv2.circle(image, mouth_left, radius=2, color=COLOR, thickness=-1)
  image = cv2.circle(image, mouth_right, radius=2, color=COLOR, thickness=-1)

  return image

def verify(target_image, threshold=10):
  """
  Perform face verification.
  """
  # Load data from database
  saved_faces, saved_faces_names = load_saved_user()
  # Perform face verification on target image, only if there is entry in the database
  if saved_faces:
    target_faces, detected_faces = get_face(target_image)
    # If there are faces detected, perform recognition on that face
    if target_faces:
      for target_face, detected_face in zip(target_faces, detected_faces):
        # Get embeddings from the target face
        target_face_embeddings = get_face_embeddings(target_face)
        # Compare the target face with all the data in database
        for every_face, name in zip(saved_faces, saved_faces_names):
          # Measure similarity with Euclidean distance
          dist = np.linalg.norm(every_face - target_face_embeddings)
          # If distance lower than threshold, the input images would be considered similar
          if dist < threshold:
            # Display the face recognition result
            print('detected_face', detected_face)
            x1, y1, x2, y2 = get_xy(detected_face['box'])
            # Draw the bounding box and keypoints on the image
            target_image = mark_face(detected_face, target_image, x1, x2, y1, y2)
            target_image = cv2.putText(target_image, name + f'_distance: {dist:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, COLOR, THICKNESS)
            break
    else:
      print('[INFO] No face detected in the image')
  return target_image

def webcam_predict(threshold=10):
    """
    Capture from webcam and perform face verification.
    """
    # open webcame and perform face recognition
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break
        
        # convert to numpy array
        image_np = np.array(frame)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # run inference on the image frame
        drawn_image = verify(image_np, threshold)
        # display result
        display_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Verification", display_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

def image_predict(image_path, THRESHOLD):
  """
  Perform face verification on .JPG files.
  """
  # Read image
  image = cv2.imread(image_path)
  # Convert RGB to BGR
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  # Perform face verification
  drawn_image = verify(image, THRESHOLD)
  # Revert image back to RGB for display
  display_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)
  # Display
  cv2.imshow("Face Verification", display_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def video_predict(video_path, THRESHOLD):
  """
  Perform face verification on .MP4 files.
  """
  # Capture the video
  cap = cv2.VideoCapture(video_path)

  # If can't capture video
  if (cap.isOpened() == False):
    print(f"[ERROR] Unable to open {video_path}")
    return
  
  # Otherwise read the image
  (success, image) = cap.read()
  startTime = 0

  while success:
    currentTime = time.time()
    fps = 1/(currentTime - startTime)
    startTime = currentTime

    # Convert RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Perform face verification
    drawn_image = verify(image, THRESHOLD)
    # Revert image back to RGB for display
    display_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)

    # put fps text on the image
    cv2.putText(display_image, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    # Show the video frame
    cv2.imshow("Face Verification", display_image)

    # Define keypress
    key = cv2.waitKey(1) & 0xFF
    
    # Pressing 'q' on the keyboard will break the loop
    if key == ord('q'):
      break

    # Otherwise, capture next frame
    (success, image) = cap.read()
  # Release everything if job is finished
  cap.release()
  cv2.destroyAllWindows()

def media_predict(test_folder, THRESHOLD):
  """
  Perform face verification on .JPG and .MP4 media files in test folder.
  """
  filenames = next(os.walk(test_folder), (None, None, []))[2]
  for f in filenames:
    extension = f.split('.')[1]
    if extension == 'jpg':
      image_predict(os.path.join(test_folder, f), THRESHOLD)

    if extension == 'mp4':
      video_predict(os.path.join(test_folder, f), THRESHOLD)


if __name__ == "main":
  
  # Set paths
  weight_path = 'facenet_keras_weights.h5'
  face_database_path = 'database'
  test_folder = 'test'

  # Global variables for color and thickness of the rectangles and keypoints that will be drawn on the image
  THRESHOLD = 10
  COLOR = (0, 255, 0)
  THICKNESS = 2

  # Initialize model and load weights
  face_detector = MTCNN()
  model = InceptionResNetV2()
  model.load_weights(weight_path)

  # Capture from webcam and predict
  webcam_predict(THRESHOLD)

  # Predict from test folder
  media_predict(test_folder, THRESHOLD)

