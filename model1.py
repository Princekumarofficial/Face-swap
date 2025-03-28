import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Read images
img1 = cv2.imread('srk.jpg')
img2 = cv2.imread('image.png')

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Function to get the landmark points given a grayscale image as input
def get_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (MediaPipe expects RGB)
    results = face_mesh.process(img_rgb)
    
    height, width, _ = img.shape
    landmarks_points = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmarks_points.append((x, y))

    points = np.array(landmarks_points, np.int32)
    return points, landmarks_points

# Function to get the convex hull of detected landmarks
def get_convexhull(img, points):
    height, width, _ = img.shape
    mask = np.zeros((height, width), np.uint8)  # Create an empty mask
    convexhull = cv2.convexHull(points)  # Compute convex hull
    return convexhull

# Get landmark points and convex hull for Image 1
img1_points, img1_landmarks = get_landmarks(img1)  # Already uses MediaPipe
img1_convex = get_convexhull(img1, img1_points)  # Compute convex hull

# Copy of image to visualize convex hull
img1_cp = img1.copy()

# Draw convex hull on the image
cv2.polylines(img1_cp, [img1_convex], isClosed=True, color=(0, 255, 255), thickness=3)

# Get landmark points and convex hull for Image 1
img2_points, img2_landmarks = get_landmarks(img2)  # Already uses MediaPipe
img2_convex = get_convexhull(img2, img2_points)  # Compute convex hull

# Copy of image to visualize convex hull
img2_cp = img2.copy()

# Draw convex hull on the image
cv2.polylines(img2_cp, [img2_convex], isClosed=True, color=(0, 255, 255), thickness=3)



