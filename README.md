Image Feature Extraction combining SIFT features and Convolutional Neural Networks
-----------------------------------------------------------------------------------
This github repository is my contributions on experimenting of SIFT feature extraction for image feature extraction combined with Convolutional Neural Networks, conducted under the mentorship of Professor Tuka Alhanai with Human-Computer Interaction lab at NYU Abu Dhabi. We try to explore the use of SIFT feature transform that employs statistical feature and then feed it into Convolutional Neural Network for the classical image classification task. 

## Goal of the project
The goal of this project was to experiment with various feature imitating network topologies for image domain. Using combined statistical feature extraction and Convolutional Neural Network, we aim to train a combined network that performs better than state-of-the-art techniques.

## Experiment Conducted
One of the main feature extraction technique we used in this project was Scale-Invariant Feature Transform (SIFT) that detects and describes local features in the image. For this, we tried various sample images and extracted their SIFT features using OpenCV's library. After that, we trained a classical Convolutional Network for 10 epochs for CIFAR-10 dataset. Then, we extracted the sift features from the images and then passed the feature-extracted images into the same network for same number of epochs and the same parameter.

## Experiment Results
The classically trained Convolutional Neural network got an accuracy of around 74% for the test set, while our SIFT-feature + CNN model got a comparable accuracy of around for the test set. This shows that our network might be able to perform as well as traditional CNN networks with a lesser training time after the SIFT features have been extracted.

## Future Work

   1.  One of the things that we are looking to improve on this project is tuning different parameters of the SIFT algorithm to optimize the feature extraction for various images and tasks. This involves using different parameters of scaling as well as keypoints for the network.
   2. Another improvement of this task would be to combine SIFT features with traditional feature extraction techniques like Speed Up Robust Features(SURF) to enhance overall representation of the images before feeding them into CNN.
   3. Finally, we can use various pre-trained models like ImageNet and VGGNet to apply transfer learning to our model. Then, we can fine-tune these models on SIFT-extracted images to utilize both generic features captured from the dataset as well as specific features captured by our SIFT algorithm.

      
 
