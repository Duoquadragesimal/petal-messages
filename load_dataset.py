train_data_path = "/dataset" # Path to your dataset

image_list = []
label_list = []

class_list_ = ["a", "b", "c"] # Our list of classifications

for category in class_list_:
    for image in os.listdir(f"{train_data_path}{category}"):
        image_list.append(f"{train_data_path}{category}/{image}")
        label_list.append(f"{category}")