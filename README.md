## Iris Recognition Computer Vision Project

For this Iris recognition project, the entire procedure follows six important steps:

Localize the Iris: In this step, we aim first to detect the pupil and then use edge detection to find the region that encloses the iris using two circles.

Iris Normalization: After locating the iris, we have an enclosed region between the circles that is the iris. The goal in this step is to convert the region, which is in polar coordinates, to rectangular images in a Cartesian plane.

Enhancement of the Normalized Image: In the first step, the image is converted to a grayscale image for better processing. However, images often have low contrast, making it challenging to see the iris patterns. Therefore, we use histogram equalization to enhance the normalized iris images, making the patterns easier to capture.

Feature Extraction: After enhancing the image, we quantify iris patterns by applying proposed filters to filter the images using two channels. We then calculate the mean and average absolute deviation of each 8x8 block on each filtered image, creating a vector containing 1536 features.

Iris Matching: With the extracted features of an iris image, we collect all iris features for our training dataset, consisting of three training images for 108 distinct persons (classes). After gathering the array containing all feature vectors for our training data, we reduce dimensions using the Fisher Linear Discriminant, experimenting with reductions from 20 up to 100 in increments of 10. Once the dimensions are reduced, we use the nearest center classifier to find the means (centers) of each class for matching test feature vectors. We use the three proposed distances (L1, L2, cosine) as distance measures to match test vectors to the closest center by returning the class that is the minimum on each of the distances.

Performance and Evaluations: Now that we have matched all test irises with a class in the training data, we create a table for dimensions up to 100 to check the CRR (correct recognition rate) for each distance, observing the best recognition. Additionally, we create a graph using the best distance. We also create a plot between FMR (false match rate) and FNMR (false non-match rate) to visualize the relationship between the number of times the process incorrectly accepts a non-match pair as a match versus how many times the process incorrectly rejects a matching pair as a non-match.

Logic of the Design:

We begin by collecting all training images in our dataset, looping through the directory localization, and finding files in each class folder, ending in .bmp. We then locate the iris and save the images with boundaries and the centers of our localized image. Following this, we use the images with boundaries and centers and normalize the image. The normalized images are then enhanced for both training and testing. For each train and testing individually, we extract the features of each enhanced image and store them in a vector. After obtaining these vectors, we use the nearest centroid classifier on the train data to find the centers for each class. Then, we calculate which center each testing feature is closest to using all three distances described in the paper. Once each testing image is classified, we find the CRR (correct recognition rate) to see which images were correctly classified to the class they belong to. We do this for different dimension reduction parameters from 20 up to 100. From this, we create a graph to visualize the different distances with all the different dimensions. Additionally, we create a table to calculate the FMR (false match rating) and FNMR (false non-match rating) and create a graph to see their correlation.


Contribution:  Simran Padam, Eric Aragundi and Llylbell
