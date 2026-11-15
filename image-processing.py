print('Importing packages...')
import numpy as np
import cv2
from scipy.signal import find_peaks

def show_image(window_name, image, max_width=600):
    height, width = image.shape[:2]

    scale = min(max_width / width, 1.0)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(window_name, resized)

def detect_origami_petals(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # 2. Convert to grayscale and apply Gaussian blur to smooth out minor paper textures
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # Smooth the image without removing too much detail
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 3. Perform Canny edge detection
    edges = cv2.Canny(blurred, 75, 200)

    # 4. Find contours from the edge-detected image
    contours, _ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter contours to isolate petals
    petal_overlay = image.copy()
    petal_count = 0
    petals = []

    for contour in contours:
        # Calculate area to filter out tiny specks or the entire flower border
        area = cv2.contourArea(contour)

        # Adjust these area bounds based on your image resolution and flower size
        if 100 < area < 10000:
            # Approximate the contour shape to smooth out jagged edges
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            #separating petals in their own images
            x, y, w, h = cv2.boundingRect(approx)

            # Keep crop coordinates inside the image
            height, width = image.shape[:2]

            x1 = max(0, x - 10)
            y1 = max(0, y - 10)
            x2 = min(width, x + w + 10)
            y2 = min(height, y + h + 10)

            petal_crop = image[y1:y2, x1:x2]

            petals.append(petal_crop)

            # Draw the detected petal contour in green
            cv2.drawContours(petal_overlay, [approx], -1, (0, 255, 0), 2)
            petal_count += 1

    print(f"Detected approximately {petal_count} petal-like regions.")

    # 6. Display the results
    #show_image('Original Image', image)
    show_image('Canny Edges', edges)
    show_image('Detected Petals', petal_overlay)
    #show_image('Petal1', petals[0])
    #show_image('Petal2', petals[1])
    #show_image('Petal3', petals[2])
    #show_image('Petal4', petals[3])
    #show_image('Petal5', petals[4])
    #show_image('Petal6', petals[5])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_flower_mask(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image")
        return

    # Convert to HSV colour space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White/pale petals have low saturation and relatively high brightness
    lower = np.array([0, 0, 100])
    upper = np.array([179, 110, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find connected regions
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No flower detected")
        return

    # Assume the flower is the largest connected region
    largest = max(contours, key=cv2.contourArea)

    flower_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(flower_mask, [largest], -1, 255, -1)

    show_image("Flower Mask", flower_mask, 500)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return flower_mask


detect_flower_mask("petalDetection.jpeg")

detect_origami_petals('petalDetection.jpeg')