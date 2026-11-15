import numpy as np
import cv2

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


def separate_flowers(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image")
        return []

    # 2. Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect pale/white flowers
    lower = np.array([0, 0, 100])
    upper = np.array([179, 110, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # 3. Remove noise and fill small gaps
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 4. Find ALL flower contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Ignore small regions
    min_area = 5000

    flower_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > min_area
    ]

    # Sort flowers from left to right
    flower_contours.sort(
        key=lambda contour: cv2.boundingRect(contour)[0]
    )

    print(f"Detected {len(flower_contours)} flowers")

    # 5. Extract each flower
    flowers = []

    for i, contour in enumerate(flower_contours):

        # Create a mask containing only this flower
        flower_mask = np.zeros(
            mask.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            flower_mask,
            [contour],
            -1,
            255,
            -1
        )

        # Remove the background
        isolated = cv2.bitwise_and(
            image,
            image,
            mask=flower_mask
        )

        # Find the flower's bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding around the flower
        padding = 20

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        # Crop the individual flower
        flower_crop = isolated[y1:y2, x1:x2]

        flowers.append(flower_crop)

        # Save the flower image
        cv2.imwrite(
            f"flower_{i+1}.png",
            flower_crop
        )

        # Display it
        show_image(
            f"Flower {i+1}",
            flower_crop,
            350
        )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return flowers
    
flowers = separate_flowers("multipleFlowers.jpeg")

print("Number of flowers:", len(flowers))