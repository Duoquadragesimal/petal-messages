print('Importing packages...')
import numpy as np
import cv2

import cv2
import numpy as np

# 1. Load the image
image_path = 'origamiFlower.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# 2. Convert to grayscale and blur to reduce noise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Apply Canny Edge Detection
# Thresholds 50 and 150 determine edge sensitivity
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# 4. Apply Probabilistic Hough Line Transform
# Parameters: image, rho, theta, threshold, minLineLength, maxLineGap
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=50,
    maxLineGap=10
)

# 5. Count the straight edges
if lines is not None:
    line_count = len(lines)
    print(f"Number of straight lines detected: {line_count}")

    # Optional: Draw the lines on the original image for visualization
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Detected Straight Edges', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No straight lines found.")