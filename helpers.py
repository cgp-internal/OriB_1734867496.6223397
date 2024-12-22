import cv2  # Import the OpenCV library for computer vision tasks
import numpy as np  # Import NumPy for numerical operations, especially with arrays


def _generate(id, size, border, output_filename="generated_aruco.png"):
    """
    Generates an ArUco marker image, adds a border, saves it, and verifies its generation.
    """
    # 1. Define the ArUco dictionary: which type of marker are we generating
    # cv2.aruco.DICT_4X4_50 is a predefined dictionary with 50 different markers of size 4x4 (a specific 'family' of markers).
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # 2. Set the marker ID: each marker in the dictionary is identified by a unique ID.
    marker_id = id

    # 3. Set the marker size in pixels: it will generate a square image of this size
    marker_size = size

    # 4. Generate the ArUco marker image: this is the core step of creating the image
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # 5. Add a border around the marker
    border_size = border
    marker_with_border = cv2.copyMakeBorder(
        marker_image,  # The source image to add border
        border_size,
        border_size,
        border_size,
        border_size,  # top, bottom, left, right border size
        cv2.BORDER_CONSTANT,  # Type of border, here a constant solid color
        value=255,  # value for constant border, here white
    )

    # 6. Define filename and save the image to disk
    filename = output_filename
    cv2.imwrite(filename, marker_with_border)

    # 7. Verify the generated marker by trying to detect it.
    verify_image = cv2.imread(
        filename, cv2.IMREAD_GRAYSCALE
    )  # load in grayscale for detection
    detector = cv2.aruco.ArucoDetector(aruco_dict)  # Initialize the ArUco detector
    corners, ids, _ = detector.detectMarkers(
        verify_image
    )  # detect the markers in the loaded image.

    if ids is not None:
        # print(f"Marker verified after generation. ID: {ids[0][0]}")
        return True

    print("Warning: Generated marker could not be verified!")

def _detect(filename):
    """
    Reads an image, detects ArUco markers, marks them with enhanced visuals, and saves the image.
    
    Args:
        filename (str): The path to the image file.
        
    Returns:
        dict: A dictionary containing marker information or None if no markers are detected.
    """
    import cv2
    import numpy as np

    print(f"Start detecting file {filename}...")
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        print("Failed to load image")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, rejected = detector.detectMarkers(gray)

    results = {}

    if ids is not None:
        # Draw the markers with standard ArUco drawing
        cv2.aruco.drawDetectedMarkers(image, corners) 

        marker_data = []
        marker_color = (0, 255, 0)  # Green color for marker
        marker_thickness = 7  # Thickness of the marker rectangle
        marker_radius = 20 # Radius of marker point
        marker_point_color = (0, 0, 255) #Color of the marker points
        marker_point_thickness = -1 # Fill the marker point. 

        for i, corner in enumerate(corners):
            corner_points = corner[0]
            top_left = corner_points[0]
            top_right = corner_points[1]
            bottom_right = corner_points[2]
            bottom_left = corner_points[3]

            center_x = int(np.mean([p[0] for p in corner_points]))
            center_y = int(np.mean([p[1] for p in corner_points]))

            # Draw the marker rectangle
            cv2.rectangle(
                image,
                (int(top_left[0]), int(top_left[1])),
                (int(bottom_right[0]), int(bottom_right[1])),
                marker_color,
                marker_thickness,
            )

            # Draw the top-left corner with circle marker
            cv2.circle(image, (int(top_left[0]), int(top_left[1])), marker_radius, marker_point_color, marker_point_thickness)
         
            dx = top_right[0] - top_left[0]
            dy = top_right[1] - top_left[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi

            marker_data.append(
                {
                    "id": int(ids[i][0]),
                    "center_position": (center_x, center_y),
                    "rotation_angle": float(f"{angle:.1f}"),
                    "corner_coordinates": {
                        "top_left": (int(top_left[0]), int(top_left[1])),
                        "top_right": (int(top_right[0]), int(top_right[1])),
                        "bottom_right": (int(bottom_right[0]), int(bottom_right[1])),
                        "bottom_left": (int(bottom_left[0]), int(bottom_left[1])),
                    },
                }
            )

        results["markers"] = marker_data
        height, width = gray.shape
        results["image_info"] = {"width": width, "height": height, "num_channels": 3}

        mask = np.ones_like(gray, dtype=np.uint8) * 255
        for corners in corners:
            corners = np.int32(corners)
            cv2.fillConvexPoly(mask, corners, 0)

        mean_val_outside = cv2.mean(gray, mask=mask)[0]
        results["outside_info"] = {"mean_gray": float(f"{mean_val_outside:.1f}")}

        cv2.imwrite("detected_aruco.png", image)
    else:
        results = None
        print("No markers detected")

    return results

# def _detect(filename):
#     """
#     Reads an image, detects ArUco markers, and returns marker information as a dictionary.

#     Args:
#         filename (str): The path to the image file.

#     Returns:
#         dict: A dictionary containing marker information or None if no markers are detected.
#     """

#     print(f"Start detecting file {filename}...")
#     image = cv2.imread(
#         filename, cv2.IMREAD_COLOR
#     )  # Read the image from disk. Load as color image for better visualization
#     if image is None:
#         print("Failed to load image")
#         return None  # If no image loaded, return None

#     gray = cv2.cvtColor(
#         image, cv2.COLOR_BGR2GRAY
#     )  # Convert color image to grayscale for marker detection

#     aruco_dict = cv2.aruco.getPredefinedDictionary(
#         cv2.aruco.DICT_4X4_50
#     )  # Use the same dictionary as the generation.
#     detector = cv2.aruco.ArucoDetector(aruco_dict)  # Init the detector

#     corners, ids, rejected = detector.detectMarkers(gray)  # Detect markers in the image

#     results = {}

#     if ids is not None:  # If markers are found
#         # Draw markers on the image (for visualization purposes)
#         cv2.aruco.drawDetectedMarkers(image, corners, ids)

#         marker_data = []  # Prepare list of markers (there may be more than 1)

#         for i, corner in enumerate(corners):  # Loop through each detected marker
#             corner_points = corner[0]  # Extract the corner points of the marker
#             top_left = corner_points[0]
#             top_right = corner_points[1]
#             bottom_right = corner_points[2]
#             bottom_left = corner_points[3]

#             center_x = int(
#                 np.mean([p[0] for p in corner_points])
#             )  # Calculate center X coordinate
#             center_y = int(
#                 np.mean([p[1] for p in corner_points])
#             )  # Calculate center Y coordinate

#             cv2.circle(
#                 image, (center_x, center_y), 4, (0, 255, 0), -1
#             )  # Draw a circle at the center

#             dx = top_right[0] - top_left[0]  # Caluclate X diff for rotation
#             dy = top_right[1] - top_left[1]  # Caluclate Y diff for rotation
#             angle = (
#                 np.arctan2(dy, dx) * 180 / np.pi
#             )  # Caluclate the angle of the marker

#             # Add each marker data to the list
#             marker_data.append(
#                 {
#                     "id": int(ids[i][0]),
#                     "center_position": (center_x, center_y),
#                     "rotation_angle": float(
#                         f"{angle:.1f}"
#                     ),  # Round to 1 decimal place for readability
#                     "corner_coordinates": {
#                         "top_left": (int(top_left[0]), int(top_left[1])),
#                         "top_right": (int(top_right[0]), int(top_right[1])),
#                         "bottom_right": (int(bottom_right[0]), int(bottom_right[1])),
#                         "bottom_left": (int(bottom_left[0]), int(bottom_left[1])),
#                     },
#                 }
#             )

#         results["markers"] = marker_data  # Store the list to the results

#         # Additional image information (assuming you want to know things about the surrounding context)
#         height, width = (
#             gray.shape
#         )  # Get the image shape (to understand the image dimensions)
#         results["image_info"] = {
#             "width": width,
#             "height": height,
#             "num_channels": (
#                 1 if len(gray.shape) == 2 else gray.shape[2]
#             ),  # Check if grayscale or color, and provide number of channels
#         }

#         # basic color analysis (average) of the area outside the marker:
#         mask = np.ones_like(gray, dtype=np.uint8) * 255  # Create a mask of all ones
#         for corners in corners:
#             corners = np.int32(corners)  # convert to int32
#             cv2.fillConvexPoly(mask, corners, 0)  # Fill the aruco in the mask to zero

#         # Average grayscale of outside
#         mean_val_outside = cv2.mean(gray, mask=mask)[0]
#         results["outside_info"] = {
#             "mean_gray": float(
#                 f"{mean_val_outside:.1f}"
#             ),  # mean gray (outside aruco area)
#             # Can add more analysis here, such as color histograms etc.
#         }

#     else:  # No markers detected
#         results = None  # No markers, so no results
#         print("No markers detected")

#     cv2.imshow('Detection Results', image) # Show the image result
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return results  # Return the generated dict or None if no markers
