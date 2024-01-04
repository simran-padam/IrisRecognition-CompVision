import os
import cv2
import numpy as np

def IrisNormalization(boundaries, centers):
    # This list will store the normalized iris images
    normalized = []

    # Loop through each boundary image and its corresponding center information
    for boundary_img, center in zip(boundaries, centers):
        # Extract the x and y coordinates of the center of the iris, and the radius of the pupil
        center_x, center_y, radius_pupil = center
        
        # The assumed width of the iris region, from the pupil boundary to the outer edge of the iris
        iris_radius = 53

        # Define the number of samples around the full circle of the iris. This determines the resolution of the angular dimension
        nsamples = 360
        # Create an array of angles from 0 to 2*pi (representing the full circle in radians)
        samples = np.linspace(0, 2*np.pi, nsamples)[:-1]
        # Initialize an array to hold the polar coordinate representation of the iris
        polar = np.zeros((iris_radius, nsamples))

        # Convert from polar to Cartesian coordinates and sample the iris texture.
        for r in range(iris_radius):
            for theta in samples:
                # Calculate Cartesian x-coordinate based on the polar coordinates and iris center
                x = int((r + radius_pupil) * np.cos(theta) + center_x)
                # Calculate Cartesian y-coordinate based on the polar coordinates and iris center
                y = int((r + radius_pupil) * np.sin(theta) + center_y)

                # Attempt to transfer the pixel value from the original image to the polar coordinate image
                try:
                    # Map the sampled point to its position in the polar image. The angle theta is scaled to the correct index
                    polar[r][int((theta * nsamples) / (2 * np.pi))] = boundary_img[y][x]
                except IndexError:
                    # If the Cartesian coordinates are outside the image bounds -- ignore this sample
                    pass

        # Normalize the polar image to have a fixed dimension for all samples (improving consistency for subsequent processing)
        # This resizes the polar representation to 64 rows (radial resolution) and 512 columns (angular resolution)
        result = cv2.resize(polar, (512, 64))
        # Add the normalized image to the list
        normalized.append(result)

    # Return the list of normalized iris images, which now have a uniform size and unwrapped circular pattern
    return normalized 
