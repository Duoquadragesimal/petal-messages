print('Importing packages...')
import numpy as np
import cv2
import apriltag
from Image_Processing import ImageProcessor as ip
print("Done!")

# 1. Load the image into the processor
my_image = ip.from_file('Image3.jpg')

# 2. Display it
my_image.show(title="Bicycle image used in the lecture");

# 3. Save a copy of it
my_image.save('Bicycle_Copy.jpg');

# 4. View the whole folder
ip.show_folder()