import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

c = 0

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transforms_train = transforms.Compose([
    transforms.ToTensor(),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform = transforms_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers = 0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform = transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers = 0)


output_dir = 'output_images1'
os.makedirs(output_dir, exist_ok=True)

# for data in trainloader:
#     images, labels = data
#     for image in images:
#         # Convert PyTorch tensor to NumPy array and permute dimensions
#         img_np = np.transpose(image.numpy(), (1, 2, 0))  # Permute (3, 32, 32) to (32, 32, 3)

#         # Create a temporary image file and save the NumPy array
#         temp_filename = os.path.join(output_dir, f'temp_temp_image_{c}.jpg')
#         cv2.imwrite(temp_filename, img_np)

#         # Read the saved image
#         img = cv2.imread(temp_filename)

#         # Remove the temporary image file
#         os.remove(temp_filename)

#         # Converting image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Applying SIFT detector
#         sift = cv2.SIFT_create()
#         kp = sift.detect(gray, None)

#         # Marking the keypoint on the image using circles
#         img_with_keypoints = cv2.drawKeypoints(
#             gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#         # Save the image with keypoints using a unique filename
#         output_filename = os.path.join(output_dir, f'image_with_keypoints_{c}.jpg')
#         cv2.imwrite(output_filename, img_with_keypoints)
#         c += 1
#         break
#     break

for data in trainloader:
    images, labels = data
    for image in images:
        # Convert PyTorch tensor to NumPy array and permute dimensions
        img_np = np.transpose(image.numpy(), (1, 2, 0))  # Permute (3, 32, 32) to (32, 32, 3)

        # Create a temporary image file and save the NumPy array
        temp_filename = os.path.join(output_dir, f'temp_temp_image_{c}.jpg')
        cv2.imwrite(temp_filename, img_np)

        # Read the saved image
        img = cv2.imread(temp_filename)

        # Remove the temporary image file
        os.remove(temp_filename)

        # Converting image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applying SIFT detector
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)

        # Marking the keypoint on the image using circles
        img_with_keypoints = cv2.drawKeypoints(
            gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display the original and SIFT image side by side using matplotlib
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with SIFT Keypoints')

        plt.show()

        # Save the image with keypoints using a unique filename
        output_filename = os.path.join(output_dir, f'image_with_keypoints_{c}.jpg')
        cv2.imwrite(output_filename, img_with_keypoints)
        c += 1
        break
    break