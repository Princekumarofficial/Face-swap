import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def initialize_face_mesh():
    """
    Initialize MediaPipe FaceMesh for landmark detection.
    
    Returns:
    mp.solutions.face_mesh.FaceMesh: Configured FaceMesh object
    """
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def display_images(img1, img2):
    """
    Display input images side by side.
    
    Args:
    img1 (numpy.ndarray): First input image
    img2 (numpy.ndarray): Second input image
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Image 1")
    
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Image 2")
    
    [axi.axis('off') for axi in axs.ravel()]
    plt.show()

def get_landmarks(img, face_mesh):
    """
    Detect facial landmarks in an image.
    
    Args:
    img (numpy.ndarray): Input image
    face_mesh (mp.solutions.face_mesh.FaceMesh): FaceMesh object
    
    Returns:
    tuple: Array of landmark points and list of landmarks
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

def match_histograms(source, target):
    source_ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    target_ycrcb = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)

    for i in range(1, 3):  # Only Cr and Cb channels (ignoring Y for brightness)
        source_hist, _ = np.histogram(source_ycrcb[:, :, i], 256, [0, 256])
        target_hist, _ = np.histogram(target_ycrcb[:, :, i], 256, [0, 256])

        cdf_source = np.cumsum(source_hist) / np.sum(source_hist)
        cdf_target = np.cumsum(target_hist) / np.sum(target_hist)

        LUT = np.interp(cdf_source, cdf_target, np.arange(256))
        source_ycrcb[:, :, i] = LUT[source_ycrcb[:, :, i]]

    return cv2.cvtColor(source_ycrcb, cv2.COLOR_YCrCb2BGR)

def get_convexhull(img, points):
    """
    Compute the convex hull of detected landmarks.
    
    Args:
    img (numpy.ndarray): Input image
    points (numpy.ndarray): Landmark points
    
    Returns:
    numpy.ndarray: Convex hull points
    """
    height, width, _ = img.shape
    mask = np.zeros((height, width), np.uint8)
    convexhull = cv2.convexHull(points)
    return convexhull

def visualize_convex_hull(img, convex_hull, title):
    """
    Visualize convex hull on an image.
    
    Args:
    img (numpy.ndarray): Input image
    convex_hull (numpy.ndarray): Convex hull points
    title (str): Plot title
    """
    img_cp = img.copy()
    cv2.polylines(img_cp, [convex_hull], isClosed=True, color=(0, 255, 255), thickness=3)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()

def perform_delaunay_triangulation(img, points, landmarks):
    """
    Perform Delaunay triangulation on facial landmarks.
    
    Args:
    img (numpy.ndarray): Input image
    points (numpy.ndarray): Landmark points
    landmarks (list): Landmark coordinates
    
    Returns:
    tuple: Triangulated image and triangle coordinates
    """
    # Get the bounding rectangle around the convex hull
    bound_rect = cv2.boundingRect(points)
    
    # Initialize Subdiv2D with bounding rectangle
    points_subdiv = cv2.Subdiv2D(bound_rect)
    points_subdiv.insert(landmarks)
    
    # Create the Delaunay triangle list
    triangles = points_subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    triangle_coords = []
    img_cp = img.copy()
    
    def get_index(arr):
        return arr[0][0] if len(arr[0]) > 0 else None
    
    # Loop to extract triangle coordinates
    for triangle in triangles:
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
        
        # Draw triangle on the image
        cv2.line(img_cp, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img_cp, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img_cp, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Get indices of landmark points for triangulation
        index_pt1 = get_index(np.where((points == pt1).all(axis=1)))
        index_pt2 = get_index(np.where((points == pt2).all(axis=1)))
        index_pt3 = get_index(np.where((points == pt3).all(axis=1)))
        
        # Append triangle if all indices are valid
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle_coords.append([index_pt1, index_pt2, index_pt3])
    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Delaunay Triangulation")
    # plt.show()
    
    return img_cp, triangle_coords

def warp_triangles(img1, img2, triangle_coords, img1_landmarks, img2_landmarks):
    """
    Warp triangles from source image to target image.
    
    Args:
    img1 (numpy.ndarray): Source image
    img2 (numpy.ndarray): Target image
    triangle_coords (list): Triangle coordinate indices
    img1_landmarks (list): Source image landmarks
    img2_landmarks (list): Target image landmarks
    
    Returns:
    numpy.ndarray: Warped image
    """
    height, width, channels = img2.shape
    img2_new_img1 = np.zeros((height, width, channels), np.uint8)
    
    for triangle in triangle_coords:
        # Get triangle points from Image 1
        pt1, pt2, pt3 = img1_landmarks[triangle[0]], img1_landmarks[triangle[1]], img1_landmarks[triangle[2]]
        
        # Bounding box around the triangle
        x, y, w, h = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
        cropped_triangle = img1[y: y+h, x: x+w]
        
        # Create a mask for the triangle
        cropped_mask = np.zeros((h, w), np.uint8)
        points = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
        cv2.fillConvexPoly(cropped_mask, points, 255)
        
        # Get triangle points from Image 2
        pt1, pt2, pt3 = img2_landmarks[triangle[0]], img2_landmarks[triangle[1]], img2_landmarks[triangle[2]]
        
        # Bounding box for Image 2 triangle
        x2, y2, w2, h2 = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
        cropped_mask2 = np.zeros((h2, w2), np.uint8)
        
        # Fill triangle mask for Image 2
        points2 = np.array([[pt1[0]-x2, pt1[1]-y2], [pt2[0]-x2, pt2[1]-y2], [pt3[0]-x2, pt3[1]-y2]], np.int32)
        cv2.fillConvexPoly(cropped_mask2, points2, 255)
        
        # Warp triangles using Affine Transform
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        dist_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))
        dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=cropped_mask2)
        
        # Combine warped triangle with Image 2
        img2_new_img1_bound_rect_area = img2_new_img1[y2: y2+h2, x2: x2+w2]
        img2_new_img1_bound_rect_area_gray = cv2.cvtColor(img2_new_img1_bound_rect_area, cv2.COLOR_BGR2GRAY)
        
        # Create inverse mask for blending
        masked_triangle = cv2.threshold(img2_new_img1_bound_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_triangle[1])
        
        # Update final blended face
        img2_new_img1_bound_rect_area = cv2.add(img2_new_img1_bound_rect_area, dist_triangle)
        img2_new_img1[y2: y2+h2, x2: x2+w2] = img2_new_img1_bound_rect_area
    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cv2.cvtColor(img2_new_img1, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Warped Face on Target Image")
    # plt.show()
    
    return img2_new_img1

def create_face_mask(img, convex_hull):
    """
    Create a mask for the face region.
    
    Args:
    img (numpy.ndarray): Input image
    convex_hull (numpy.ndarray): Convex hull points
    
    Returns:
    tuple: Head mask and inverse mask
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mask = np.zeros_like(img_gray)
    
    # Create a filled convex mask using landmarks
    head_mask = cv2.fillConvexPoly(img_mask, convex_hull, 255)
    
    # Create an inverse mask to remove face region
    img_mask = cv2.bitwise_not(head_mask)
    
    return head_mask, img_mask

def blend_faces(img1, img2, warped_face, head_mask):
    """
    Blend the warped face with the target image.
    
    Args:
    img1 (numpy.ndarray): Source image
    img2 (numpy.ndarray): Target image
    warped_face (numpy.ndarray): Warped face from source image
    head_mask (numpy.ndarray): Head region mask
    
    Returns:
    numpy.ndarray: Blended image
    """
    # Remove the original face from Image 2
    img2_maskless = cv2.bitwise_and(img2, img2, mask=cv2.bitwise_not(head_mask))
    
    # Blend the new swapped face with the original image
    result = cv2.add(img2_maskless, warped_face)
    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Final Face Swap Result")
    # plt.show()
    
    return result

def seamless_clone_face(img1, img2, result, head_mask):
    """
    Perform seamless cloning to blend the face naturally.
    
    Args:
    img1 (numpy.ndarray): Source image
    img2 (numpy.ndarray): Target image
    result (numpy.ndarray): Initial blended result
    head_mask (numpy.ndarray): Head region mask
    
    Returns:
    numpy.ndarray: Seamlessly blended image
    """
    # Get the bounding box around the convex hull of the swapped face
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(head_mask))
    
    # Compute the face center for seamless cloning
    face_center = (int(x + w / 2), int(y + h / 2))
    
    # Perform seamless cloning
    # Convert head_mask to 3-channel image
    head_mask_3channel = cv2.merge([head_mask, head_mask, head_mask])
    
    seamlessclone = cv2.seamlessClone(result, img2, head_mask_3channel, face_center, cv2.NORMAL_CLONE)
    
    output_path = "output_face_swap.jpg"
    cv2.imwrite(output_path, seamlessclone)
    return output_path

def main_face_swap(img1, img2):
    """
    Main function to perform face swapping.
    
    Args:
    img1 (numpy.ndarray): Source image with face to be swapped
    img2 (numpy.ndarray): Target image to receive the face
    
    Returns:
    numpy.ndarray: Final face-swapped image
    """
    # Initialize face mesh
    face_mesh = initialize_face_mesh()
    
    # Display input images
    # display_images(img1, img2)
    
    # Get landmarks for both images
    img1_points, img1_landmarks = get_landmarks(img1, face_mesh)
    img2_points, img2_landmarks = get_landmarks(img2, face_mesh)
    
    # Get convex hull for both images
    img1_convex = get_convexhull(img1, img1_points)
    img2_convex = get_convexhull(img2, img2_points)
    
    # Visualize convex hull
    # visualize_convex_hull(img1, img1_convex, "Convex Hull on Image 1")
    # visualize_convex_hull(img2, img2_convex, "Convex Hull on Image 2")
    
    # Perform Delaunay triangulation
    _, triangle_coords = perform_delaunay_triangulation(img1, img1_points, img1_landmarks)
    
    # Warp triangles
    warped_face = warp_triangles(img1, img2, triangle_coords, img1_landmarks, img2_landmarks)
    
    # Create face masks
    head_mask, _ = create_face_mask(img2, img2_convex)
    
    # Blend faces
    result = blend_faces(img1, img2, warped_face, head_mask)
    
    # Perform seamless cloning
    final_result = seamless_clone_face(img1, img2, result, head_mask)

    print("Face swap completed successfully!")
    print("Output saved as:", final_result)
    
    return final_result

# Example usage
if __name__ == "__main__":
    # Assume img1 and img2 are loaded images
    img1 = cv2.imread('srk.jpg')
    img2 = cv2.imread('image.png')
    result = main_face_swap(img1, img2)
    pass