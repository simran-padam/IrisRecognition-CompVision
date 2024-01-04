import pickle
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import IrisEnhancement
import IrisFeatureExtraction
import IrisLocalization
import IrisNormalization
import IrisMatching
import IrisPerformanceEvaluation
import pickle
import statistics

number_of_classes = 108
images_per_train_class  = 3
images_per_test_class = 3

with open('images_features_train.pkl', 'rb') as file:
    images_features_train = pickle.load(file)

with open('images_features_test.pkl', 'rb') as file:
    images_features_test = pickle.load(file)

    #SIMRAN TESTING

def main():

    number_of_classes = 108
    images_per_train_class  = 3
    images_per_test_class = 4

    crr_d1 = []
    crr_d2 = [] 
    crr_d3 = []
    dims = []

    thresholds = [0.446, 0.472, 0.502]
    fmr = []
    fnmr = []

    train_labels = np.repeat(np.arange(1,number_of_classes+1),images_per_train_class)
    test_labels = np.repeat(np.arange(1,number_of_classes+1),images_per_test_class) 
    index = np.arange(1,len(images_features_train)+1)
    data = {'train_labels': train_labels, 'index': index}
    train_label_df = pd.DataFrame(data)

    assert len(train_labels) == len(images_features_train)

    for i in range(20,110, 10):
        images_features_dr_train, images_features_dr_test= IrisMatching.dimension_reduction(images_features_train, images_features_test, train_labels, k = i)

        d1 = IrisMatching.nearestCentroid(images_features_dr_train,images_features_dr_test ,train_labels, metric = 'l1', score = False)
        d2 = IrisMatching.nearestCentroid(images_features_dr_train,images_features_dr_test ,train_labels, metric = 'l2', score = False )
        d3 = IrisMatching.nearestCentroid(images_features_dr_train,images_features_dr_test ,train_labels, metric = 'cosine', score =  False)

        assert len(d1) == number_of_classes * images_per_test_class 
        assert len(d2) == number_of_classes * images_per_test_class 
        assert len(d3) == number_of_classes * images_per_test_class 

        #correct recognition rate 
        dims.append(i)
        crr_d1.append(IrisPerformanceEvaluation.CRR(test_labels,d1))
        crr_d2.append(IrisPerformanceEvaluation.CRR(test_labels,d2))
        crr_d3.append(IrisPerformanceEvaluation.CRR(test_labels,d3))

        similarity_score = IrisMatching.nearestCentroid(images_features_dr_train,images_features_dr_test ,train_labels, metric = 'cosine', score =  True)

        df1 = IrisPerformanceEvaluation.false_rate(similarity_score, thresholds[0], test_labels, d3)
        df2 = IrisPerformanceEvaluation.false_rate(similarity_score, thresholds[1], test_labels, d3)
        df3 = IrisPerformanceEvaluation.false_rate(similarity_score, thresholds[2], test_labels, d3)

        false_rate_table = pd.concat([df1,df2,df3])
        print(false_rate_table)

    crr_data = {'dims':dims,'crr_d1':crr_d1,'crr_d2':crr_d2,'crr_d3':crr_d3}
    crr_df = pd.DataFrame(crr_data)
    
    print(crr_df)


    IrisPerformanceEvaluation.make_plot(crr_df)

if __name__ == "__main__":
  main()




   





