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

# Detect landmarks in the input image and extract the face region
def extract_face_region(face_mesh, image_path):
    image, rgb_image = load_face_image(image_path)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        x_min, y_min, x_max, y_max = image.shape[1], image.shape[0], 0, 0
        for landmark in results.multi_face_landmarks[0].landmark:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)
        face_crop = image[y_min:y_max, x_min:x_max]
        return face_crop
    return None

# Overlay the face onto video feed
def overlay_face_on_video(face_mesh, face_crop, video_source=0):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks and face_crop is not None:
            ih, iw, _ = frame.shape
            for landmark in results.multi_face_landmarks[0].landmark:
                # Calculate bounding box for detected face
                x_min, y_min, x_max, y_max = iw, ih, 0, 0
                for lm in results.multi_face_landmarks[0].landmark:
                    lx, ly = int(lm.x * iw), int(lm.y * ih)
                    x_min, y_min = min(x_min, lx), min(y_min, ly)
                    x_max, y_max = max(x_max, lx), max(y_max, ly)

                # Scale face_crop to match detected face size
                face_width = x_max - x_min
                face_height = y_max - y_min
                scaled_face_crop = cv2.resize(face_crop, (face_width, face_height))

                # Calculate rotation angle based on face landmarks
                left_eye = results.multi_face_landmarks[0].landmark[33]  # Example: left eye landmark
                right_eye = results.multi_face_landmarks[0].landmark[263]  # Example: right eye landmark
                dx = (right_eye.x - left_eye.x) * iw
                dy = (right_eye.y - left_eye.y) * ih
                angle = -cv2.fastAtan2(dy, dx)

                # Rotate the face_crop
                center = (scaled_face_crop.shape[1] // 2, scaled_face_crop.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_face_crop = cv2.warpAffine(scaled_face_crop, rotation_matrix, (scaled_face_crop.shape[1], scaled_face_crop.shape[0]))

                # Overlay rotated face_crop onto the frame
                y_start = max(0, y_min)
                y_end = min(ih, y_min + face_height)
                x_start = max(0, x_min)
                x_end = min(iw, x_min + face_width)
                frame[y_start:y_end, x_start:x_end] = rotated_face_crop[:y_end - y_start, :x_end - x_start]
                break

        cv2.imshow("Face Overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_mesh = initialize_face_mesh()
    face_crop = extract_face_region(face_mesh, "srk.jpg")
    overlay_face_on_video(face_mesh, face_crop, video_source=0)
