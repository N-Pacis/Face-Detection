import cv2
import os

# Load the group photo
img = cv2.imread('group_photo.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load OpenCV's face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Create a directory to save the cropped faces
if not os.path.exists('cropped_faces'):
    os.makedirs('cropped_faces')

# Loop through each detected face
for i, (x, y, w, h) in enumerate(faces):
    # Crop the face from the original image
    crop_img = img[y:y+h, x:x+w]
    # Save the cropped face as a separate image
    cv2.imwrite('cropped_faces/face_' + str(i) + '.jpg', crop_img)
