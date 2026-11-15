from separate_flowers import separate_flowers
from predict import predict_petal
from image_processing_test import detect_origami_petals

flower_rows = separate_flowers("multipleFlowers.jpeg")

for row_number, row in enumerate(flower_rows, start=1):
    for flower_number, flower in enumerate(row, start=1):
        
        petals = detect_origami_petals(flower)

        if len(petals) != 5:
            print(f"Could not process flower {flower_number} in row {row_number}")
            continue

        binary = ""

        for petal in petals:
            prediction = predict_petal(petal)

            if prediction == "folded-petal":
                binary += "1"
            else:
                binary += "0"

        print(f"Row {row_number}, Flower {flower_number}: {binary}")
        