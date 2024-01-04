import cv2
import numpy as np
import math
from scipy.spatial import distance
import os

def IrisLocalization(images):

    # Step 1: Initialize lists to store results
    boundaries = [] 
    iris_centers = []

    i = 0
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply Bilateral Filter to reduce noise and maintain edges
        smoothed = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # Step 3: Estimate pupil center using image projections
        h_proj = np.mean(smoothed, 0)
        v_proj = np.mean(smoothed, 1)
        tentative_center_x = h_proj.argmin()
        tentative_center_y = v_proj.argmin()
        
        # Step 4: Fine-tune pupil center estimate within a smaller region
        local_region_x = smoothed[tentative_center_x-60:tentative_center_x+60]
        local_region_y = smoothed[tentative_center_y-60:tentative_center_y+60]


        h_proj_local = np.mean(local_region_y, 0)
        v_proj_local = np.mean(local_region_x, 0)
        refined_center_x = h_proj_local.argmin()
        refined_center_y = v_proj_local.argmin()

        # Step 5: Threshold to focus on darker regions (pupil)
        masked = cv2.inRange(smoothed, 0, 70)
        masked_img = cv2.bitwise_and(smoothed, masked)
        edges = cv2.Canny(masked_img, 100, 220)
        
        # Step 6: Apply Hough Transform to detect circles
        potential_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 5, 100)
        
        center_estimate = (refined_center_x, refined_center_y)

        closest_circle = min(potential_circles[0], key=lambda x: distance.euclidean(center_estimate, (x[0], x[1])))
        
        final_center_x = int(closest_circle[0])
        final_center_y = int(closest_circle[1])
        radius = int(closest_circle[2])

        # Step 7: Draw detected boundaries
        cv2.circle(gray_image, (final_center_x, final_center_y), radius, (255, 255, 0), 1)
        cv2.circle(gray_image, (final_center_x, final_center_y), radius + 53, (255, 255, 0), 1)



        i +=1
        output_directory = "./localized_images/"

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_path = os.path.join(output_directory, f"localized_{i}.png")
        cv2.imwrite(output_path, gray_image)
        
        boundaries.append(gray_image)
        iris_centers.append([final_center_x, final_center_y, radius])

    return boundaries, iris_centers

