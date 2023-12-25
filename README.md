# Image preprocessing techniques in Machine learning 
- We're looking at 4 image preprocessing techniques 
* Grayscale

Grayscale code is below
```python
import os
from PIL import Image

# Your image path
image_path = r'C:\Users\Jimi2\Documents\Test\Healthy_test\122.png'  # Ensure the 'r' before the path to avoid escape characters

# Construct the absolute path from the notebook's directory
notebook_directory = r'C:\Users\Jimi2\Documents\preprocessing'
absolute_image_path = os.path.join(notebook_directory, image_path)

# Load the image using PIL
image = Image.open(absolute_image_path)

# Display the image if needed
# image.show()

# Import necessary libraries
# from PIL import Image
import matplotlib.pyplot as plt

# Load the image
# image_path = 'path_to_your_image.jpg'  # Replace with your image path
# image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Display original and grayscale images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

plt.tight_layout()
plt.show()
```

- Result is below:
## This preprocessing image technique is called color correction
![Grayscaled image](../grayscaled%20image%20example.png)



# Thresholding 
## Pre-processing technique is called segmentation

```python
import cv2
import os

# Path to the image folder and where you want to save thresholded images
image_folder = r'C:\Users\Jimi2\Documents\Test\Unhealthy_test'
output_folder = r'C:\Users\Jimi2\Documents\preprocessing\Thresholding\Threshold_unhealthy_test'

# Loop through images
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale

    # Apply thresholding (example with simple threshold)
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Save thresholded images
    output_path = os.path.join(output_folder, f'healthy_{img_name}')
    cv2.imwrite(output_path, thresholded)
```

# Noise Reduction
## Pre processing technique is noise reduction
```python
import cv2
import os

# Define paths
code_file_path = r'C:\Users\Jimi2\Documents\preprocessing\Noise Reduction'
image_folder_path = r'C:\Users\Jimi2\Documents\Test\Healthy_test'
output_folder_path = r'C:\Users\Jimi2\Documents\preprocessing\Noise Reduction\NoiseReduced_healthy_test'

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Loop through images in the folder
for img_name in os.listdir(image_folder_path):
    img_path = os.path.join(image_folder_path, img_name)
    
    # Read the image
    img = cv2.imread(img_path)
    
    # Apply noise reduction (denoising)
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    
    # Save the denoised image with the title "healthy" in the output folder
    output_img_path = os.path.join(output_folder_path, f'healthy_{img_name}')
    cv2.imwrite(output_img_path, denoised_img)
```