import cv2
import numpy as np
from scipy import ndimage

# Function to create a filter given a channel dx and dy, crop_amount was to define a cropped grid to test 
def create_filter(dx,dy,crop_amount):

    # Defining the frequency
    f = 1/dy


    # Grids for the 8x8 kernel to be able to apply the filter using convolution later
    # x, y = np.mgrid[-7:8, -7:8]
    # x, y = np.mgrid[0:crop_amount, 0:512]
    x, y = np.mgrid[0:8, 0:8]

    #Filter provided in the paper
    filter = (1/(2*np.pi*dx*dy))*np.exp(-(x**2 / (2*dx**2) + y**2 / (2*dy**2))) * np.cos(2 * np.pi * f * (x**2 + y**2)**0.5)
    
    return filter


#Funtion to split my filtered image in equal amount of 8x8 blocks
def split_blocks(image, block_size=(8,8)):

  #calculating amount of squares I could have depending on the size
  rows = image.shape[0]//block_size[0]
  cols = image.shape[1]//block_size[1]

  # loop to split the blocks and save them into the blocks array
  blocks = []
  for row in range(rows):
    for col in range(cols):
      block = image[row*block_size[0]:(row+1)*block_size[0], 
                    col*block_size[1]:(col+1)*block_size[1]]
      blocks.append(block)

  return blocks


# Principal function to extract the features.
def feature_extraction(image,crop_amount):

    # array to saved each mean and abs deviation for each block
    feature_vectors = []

    

    # Define filters kernels using the function to create the filters from the paper
    filter1 = create_filter(3,1.5,crop_amount)
    filter2 = create_filter(4.5,1.5,crop_amount)

    #Convolve each filter on the image to have 2 filtered images with both channels
    filtered1 = cv2.filter2D(image, -1, filter1)
    filtered2 = cv2.filter2D(image, -1, filter2)

    #split both images on equal 8x8 blocks
    blocks_image1 = split_blocks(filtered1)
    blocks_image2 = split_blocks(filtered2)

    
    # For each block on each filtered image, calculate the mean and the avg abs deviation 
    # to append to the feature array, first loop is for the blocks in filtered image 1
    for block in blocks_image1:
        block_mean = np.mean(block,axis=(0,1))
        absolute_deviations = np.abs(block - block_mean)
        aad = np.mean(absolute_deviations, axis=(0, 1))

        feature_vectors.append(block_mean)
        feature_vectors.append(aad)

    # this loop is to calculate the mean and avg abs deviaiton on the filtered image 2
    for block in blocks_image2:
        block_mean = np.mean(block,axis=(0,1))
        absolute_deviations = np.abs(block - block_mean)
        aad = np.mean(absolute_deviations, axis=(0, 1))

        feature_vectors.append(block_mean)
        feature_vectors.append(aad)


    # print(len(blocks_image1))


    return feature_vectors

    
