import cv2
import numpy as np
def detect_origami_petals(image):
    if image is None or image.size == 0:
        print("Error: Invalid image.")
        return []

    # 2. Convert to grayscale and apply Gaussian blur to smooth out minor paper textures
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 4)

    # 3. Perform Canny edge detection
    edges = cv2.Canny(blurred, 40, 150)

    # 4. Find contours from the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter contours to isolate petals
    petal_overlay = image.copy()
    petal_count = 0
    petals = []
    petal_coords = []

    for contour in contours:
        # Calculate area to filter out tiny specks or the entire flower border
        area = cv2.contourArea(contour)

        # Adjust these area bounds based on your image resolution and flower size
        if 30 < area < 400:
            # Approximate the contour shape to smooth out jagged edges
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            #separating petals in their own images
            x, y, w, h = cv2.boundingRect(approx)
            petal_crop = image[y-10:y + h + 10, x-10:x + w + 10]

            if w > 150 or h > 150:

                if len(petals) == 0:
                    petals.append(petal_crop)
                    petal_coords.append([x, y])

                    # Draw the detected petal contour in green
                    cv2.drawContours(petal_overlay, [approx], -1, (0, 255, 0), 2)
                    petal_count += 1

                else:
                    duplicate = False
                    for coords in petal_coords:
                        x1, y1 = coords
                        if (x1 - x < 20) & (y1 - y < 20) == True:
                            duplicate = True

                    if duplicate == False:
                        petals.append(petal_crop)
                        petal_coords.append([x, y])

                        # Draw the detected petal contour in green
                        cv2.drawContours(petal_overlay, [approx], -1, (0, 255, 0), 2)
                        petal_count += 1

    #order the petals
    if len(petals) != 5:
        print(f"Expected 5 petals, detected {len(petals)}")
        return []

    combined_list = []
    for i in range(5):
        combined_list.append((petal_coords[i][0],petal_coords[i][1], petals[i]))

    for i in range(5):
        swapped = False
        for j in range(0, 5 - i - 1):
            if combined_list[j][1] > combined_list[j + 1][1]:
                combined_list[j], combined_list[j + 1] = combined_list[j + 1], combined_list[j]
                swapped = True
        if not swapped:
            break
    if combined_list[1][0] > combined_list[2][0]:
        combined_list[1], combined_list[2] = combined_list[2], combined_list[1]
    if combined_list[3][0] > combined_list[4][0]:
        combined_list[3], combined_list[4] = combined_list[4], combined_list[3]

    # Extract the petal images in their new order
    petals = [item[2] for item in combined_list]
    print(f"Detected approximately {petal_count} petal-like regions.")

    # 6. Display the results
    #cv2.imshow('Original Image', image)
    cv2.imshow('Canny Edges', edges)
    cv2.imshow('Detected Petals', petal_overlay)
    cv2.imshow('Petal1', petals[0])
    cv2.imshow('Petal2', petals[1])
    cv2.imshow('Petal3', petals[2])
    cv2.imshow('Petal4', petals[3])
    cv2.imshow('Petal5', petals[4])
    #cv2.imshow('Petal6', petals[5])
    #cv2.imshow('Petal7', petals[6])
    #cv2.imshow('Petal8', petals[7])
    #cv2.imshow('Petal9', petals[8])

    cv2.destroyAllWindows()
    return petals


if __name__ == "__main__":
    image = cv2.imread("flowerCopy.png")
    detect_origami_petals(image)