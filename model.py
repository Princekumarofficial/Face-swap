import numpy as np
import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Load input face image
def load_face_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, rgb_image

# Initialize Face Mesh
def initialize_face_mesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5):
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence
    )

# Perform Delaunay triangulation on detected face landmarks
def perform_delaunay_triangulation(image, landmarks):
    height, width, _ = image.shape
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)

    # Insert landmarks into the subdiv
    for landmark in landmarks:
        subdiv.insert((landmark.x * width, landmark.y * height))

    # Get the list of triangles
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # Draw the triangles on the image
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            cv2.line(image, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(image, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(image, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA)

    return image

# Helper function to check if a point is inside a rectangle
def rect_contains(rect, point):
    return rect[0] <= point[0] < rect[2] and rect[1] <= point[1] < rect[3]

# Modify extract_face_region to include triangulation
def extract_face_region_with_triangulation(face_mesh, image_path):
    image, rgb_image = load_face_image(image_path)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        triangulated_image = perform_delaunay_triangulation(image.copy(), landmarks)
        return triangulated_image
    return None

if __name__ == "__main__":
    face_mesh = initialize_face_mesh()
    triangulated_face = extract_face_region_with_triangulation(face_mesh, "srk.jpg")
    if triangulated_face is not None:
        cv2.imshow("Delaunay Triangulation", triangulated_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
