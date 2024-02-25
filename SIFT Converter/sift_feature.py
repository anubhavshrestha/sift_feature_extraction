
# # Important NOTE:  Use opencv <= 3.4.2.16 as
# # SIFT is no longer available in
# # opencv > 3.4.2.16
import cv2

# Loading the image
file_path = 'dog_playing.jpeg'
img = cv2.imread(file_path)



# Converting image to grayscale
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.imshow(img)
# Applying SIFT detector
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
 
# Marking the keypoint on the image using circles
img=cv2.drawKeypoints(gray ,
                      kp ,
                      img ,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
cv2.imwrite('image-with-keypoints1.jpg', img)

'''Writes in a image file'''
# import cv2
# import numpy as np

# # Load the image
# image_path = 'sift_feature_img2.png'  # Replace with the path to your image
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Initialize the SIFT detector
# sift = cv2.SIFT_create()

# # Detect SIFT keypoints and compute descriptors
# keypoints, descriptors = sift.detectAndCompute(img, None)

# # Save the descriptors to a file
# output_file = 'sift_features.txt'  # Choose a filename for saving the descriptors
# np.savetxt(output_file, descriptors, delimiter=',', fmt='%1.5f')

# print(f'SIFT descriptors saved to {output_file}.')

# sift_features_file = 'sift_features_txt'
# sift_features_matrix = np.loadtxt(sift_features_file, delimiter = ',')
# print(sift_features_matrix)

'''This is the function that does in a folder'''
# import cv2
# import numpy as np
# import os

# # Set the path to the folder containing your images
# image_folder = 'images'  # Replace with the path to your image folder

# # Initialize the SIFT detector
# sift = cv2.SIFT_create()

# # Initialize lists to store the SIFT features and image numbers
# sift_features_list = []
# image_numbers = []

# # Process each image in the folder
# for image_number, filename in enumerate(os.listdir(image_folder)):
#     if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # You can add more extensions if needed
#         # Read the image
#         img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
#         print(img.shape)

#         # Detect SIFT keypoints and compute descriptors
#         keypoints, descriptors = sift.detectAndCompute(img, None)

#         # Create a 3D vector for each SIFT descriptor
#         sift_vectors = np.hstack((np.ones((descriptors.shape[0], 1)) * image_number, descriptors))

#         # Append the sift_vectors to the list
#         sift_features_list.append(sift_vectors)

#         # Store the image number
#         image_numbers.append(image_number)

# # Concatenate the SIFT features and image numbers into a single NumPy array
# sift_features = np.vstack(sift_features_list)
# image_numbers = np.array(image_numbers)






# Now, sift_features contains the 3D vectors where the first dimension is the image number,
# and the other two dimensions represent the SIFT descriptor.

# Example: Access the SIFT feature for the 5th image and the 10th descriptor
# image_index = 5
# descriptor_index = 10
# feature = sift_features[(image_numbers == image_index) & (sift_features[:, 0] == descriptor_index)]

# You can use this sift_features array for further analysis or processing.
