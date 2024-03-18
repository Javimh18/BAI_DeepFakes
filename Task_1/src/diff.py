from PIL import Image
import os
import numpy as np

path_real_recons = "/home/javiermunoz/Universidad/MasterDeepLearning/BAI/Practicas/BAI_DeepFakes/Task_1/results/vae/generated/fake/0001_fake/fake.000312.jpg"
path_fake_recons = "/home/javiermunoz/Universidad/MasterDeepLearning/BAI/Practicas/BAI_DeepFakes/Task_1/results/vae/generated/real/0001/000293.jpg"

# Load the two RGB images
image1 = np.array(Image.open(path_real_recons))
image2 = np.array(Image.open(path_fake_recons))

# Ensure images have the same shape
if image1.shape != image2.shape:
    raise ValueError("Images must have the same shape.")

# Compute the absolute difference between the images
difference = np.abs(image1 - image2)

# Optionally, you can convert the difference to grayscale for visualization
difference_gray = np.mean(difference, axis=2).astype(np.uint8)

print(difference_gray)

difference_image = Image.fromarray(difference_gray)
difference_image.save("difference_image.jpg")
