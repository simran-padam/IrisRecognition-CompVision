import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#correct recognition rate function
def CRR(true, pred): 
    correct = 0
    n = len(true)
    
    #for each pair of true and pred, function increments correct counter by 1
    for i,j in zip(true, pred):
        if i == j :
            correct +=1
        else:
            correct +=0
    #Correct recognition rate in percentage is obtained by dividing number of correct and total number of vectors
    CRR = np.round(np.divide(correct, n)*100,2)

    return CRR


#false match rate calculation
def false_rate(similarity_score, threshold, test_labels, d3):

    #combine key columns to calculate FMR and FNMR
    false_data = {'similarity_score': similarity_score, 'test_labels': test_labels, 'cosine_match': d3}
    false_df = pd.DataFrame(false_data)

    #False Match: if score < threshold and test == match (we should reject but got accepted)
    false_df['FMR'] = false_df.apply(lambda x: 1 if (x['similarity_score'] < threshold and x['test_labels'] == x['cosine_match']) else 0, axis=1)
    #False Non-Match: if score > threshold and test != match (we should accept but got rejected)
    false_df['FNMR'] = false_df.apply(lambda x: 1 if (x['similarity_score'] > threshold  and x['test_labels'] != x['cosine_match']) else 0, axis=1)
    false_df['threshold'] = threshold
    false_df = false_df.groupby(["threshold"], as_index = False)["FMR", "FNMR"].mean()

    return false_df

#function to create CRR plot for all dimensions using matplotlib
def make_plot(df):
    plt.plot( df["dims"], df["crr_d3"],marker='*', markersize=10)
    plt.title('Recognition results using features of different dimensionality')
    plt.ylabel('Correct Recognition Rate')
    plt.xlabel('Number of dimensions')
    plt.show()


    







