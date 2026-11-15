print('Importing packages...')
import numpy as np
import cv2

def detect_origami_petals(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # 2. Convert to grayscale and apply Gaussian blur to smooth out minor paper textures
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    # 3. Perform Canny edge detection
    edges = cv2.Canny(blurred, 40, 200)

    # 4. Find contours from the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter contours to isolate petals
    petal_overlay = image.copy()
    petal_count = 0
    petals = []

    for contour in contours:
        # Calculate area to filter out tiny specks or the entire flower border
        area = cv2.contourArea(contour)

        # Adjust these area bounds based on your image resolution and flower size
        if 40 < area < 200:
            # Approximate the contour shape to smooth out jagged edges
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            #separating petals in their own images
            x, y, w, h = cv2.boundingRect(approx)
            petal_crop = image[y-10:y + h + 10, x-10:x + w + 10]
            petals.append(petal_crop)

            # Draw the detected petal contour in green
            cv2.drawContours(petal_overlay, [approx], -1, (0, 255, 0), 2)
            petal_count += 1

    print(f"Detected approximately {petal_count} petal-like regions.")

    # 6. Display the results
    #cv2.imshow('Original Image', image)
    cv2.imshow('Canny Edges', edges)
    cv2.imshow('Detected Petals', petal_overlay)
    #cv2.imshow('Petal1', petals[0])
    #cv2.imshow('Petal2', petals[1])
    #cv2.imshow('Petal3', petals[2])
    #cv2.imshow('Petal4', petals[3])
    #cv2.imshow('Petal5', petals[4])
    #cv2.imshow('Petal6', petals[5])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_origami_petals('petalDetection.jpg')