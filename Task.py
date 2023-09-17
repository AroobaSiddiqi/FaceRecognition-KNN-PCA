import numpy as np
import csv
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import distance


"""-------------------------------------------------------------DATASET PROCESSING------------------------------------------------------------"""

def read_csv(file,num_classes):

    with open(file, 'r') as file:
        reader = csv.reader(file)
        dataset = [np.array(list(map(float, row))) for row in reader] #list of lists
        dataset = dataset[:(num_classes*170)]
        
        return dataset

def normalize_dataset(nested_list): 

    normalized_dataset = []
    
    for vector in nested_list:
        magnitude = np.linalg.norm(vector)  # Compute the magnitude of the vector
        normalized_vector = (vector/magnitude).tolist()  # Divide each element by the magnitude
        normalized_dataset.append(normalized_vector)
    
    return normalized_dataset

def label_dataset(nested_list):
    
    for i, inner_list in enumerate(nested_list):
        label = (i // 170) + 1  # Calculate the label based on the pattern
        inner_list.append(label)  # Add the label as the last element in the inner list
    
    return nested_list

def train_test_split(nested_list,num_train):

    train = []
    test = []

    for i in range(0, len(nested_list), 170):
        batch = nested_list[i:i+170]  # Get the next batch of 170 inner lists
        random.shuffle(batch)  # Shuffle the batch in-place
        test.extend(batch[num_train:])  # Add the first 20 to the test list
        train.extend(batch[:num_train])  # Add the remaining 150 to the train list

    return train, test

def process_dataset(csv,num_classes):
    
    dataset = read_csv(csv,num_classes)
    normalized = normalize_dataset(dataset)
    labelled = label_dataset(normalized)
    
    return labelled

"""---------------------------------------------------------------KNN CLASSIFIER--------------------------------------------------------------"""

def get_inv_cov_matrix(nested_list):

    pixel_data = [x[:-1] for x in nested_list]

    cov_matrix = np.cov(pixel_data.T)
    inv_cov_matrix  = np.linalg.inv(cov_matrix)

    return inv_cov_matrix

def mahalanobis_distance(vector1, vector2, inv_cov_matrix):

    mahalanobis_dist = distance.mahalanobis(vector1, vector2, inv_cov_matrix)
    return mahalanobis_dist

def euclidean_distance(vector1, vector2):

    euclidean_distance = distance.euclidean(vector1, vector2)
    return euclidean_distance

def cosine_similarity(vector1, vector2):

    cosine_similarity = np.dot(vector1, vector2)
    return cosine_similarity

def get_distance(vector1, vector2, metric, *inv_cov_matrix):

    if metric == 'euclidean':
        return euclidean_distance(vector1, vector2)
    elif metric == 'mahalanobis':
        return mahalanobis_distance(vector1, vector2, inv_cov_matrix[0])
    elif metric == 'cosine':
        return cosine_similarity(vector1, vector2)
    else:
        raise ValueError("Invalid distance metric")

def get_neighbors(training_set, test_instance, k, metric):
    
    distances = [] # list of tuples
    vector1 = vector2 = None

    if metric == 'mahalanobis':
        inv_cov_matrix  = get_inv_cov_matrix(training_set)
        parameters = [vector1, vector2, metric, inv_cov_matrix]
    else:
        parameters = [vector1, vector2, metric]

    for i in range(len(training_set)):
        
        vector1 = training_set[i][:-1]
        vector2 = test_instance[:-1]

        parameters[0:2] = [vector1, vector2]

        distance = get_distance(*parameters)
            
        label = training_set[i][-1]
        distances.append((label,distance))

    if metric=='cosine':
        distances.sort(key=lambda x: -x[1])  # similarity is sorted in descending order
    else:
        distances.sort(key=lambda x: x[1])  # distance is sorted in ascending order

    # list of tuples containing (label,distance) from k nearest neighbours
    neighbors = [] 
    for j in range(k):
        neighbors.append(distances[j][0])

    return neighbors

def get_majority_vote(neighbors):

    votes = {}

    for neighbor in neighbors:
        if neighbor in votes:
            votes[neighbor] += 1
        else:
            votes[neighbor] = 1
    majority_vote = max(votes, key=votes.get)

    return majority_vote

def kNN_classifier(training_set, test_set, k, metric):
    

    predictions = []
   
    for test_instance in test_set:
        neighbors = get_neighbors(training_set, test_instance, k, metric)
        majority_vote = get_majority_vote(neighbors)
        label = test_instance[-1]
        predictions.append((majority_vote,label)) #(predicted,actual)
   
    return predictions

def get_accuracy(nested_list):
    
    correct = 0
    
    for prediction in nested_list:

        if prediction[0]==prediction[1]:
            correct+=1

    accuracy = (correct/len(nested_list))*100
    return accuracy

def kNN_performance(csv='fea.csv',k=1,metric='euclidean',num_classes=10,num_train=150):

    total_time = 0
    total_accuracy = 0
    dataset = process_dataset(csv,num_classes)

    for _ in range(5):

        train, test = train_test_split(dataset,num_train)

        start = time.time()

        predictions = kNN_classifier(train,test,k,metric)
        end = time.time()

        current_time = end - start

        total_time += current_time

        current_accuracy = get_accuracy(predictions)
        total_accuracy += current_accuracy

    avg_accuracy = round(total_accuracy/5,1)
    avg_time = round(total_time/5,1)

    return avg_accuracy, avg_time

"""--------------------------------------------------------------------PCA--------------------------------------------------------------------"""

def pca_performance(n_components):

    dataset = process_dataset(csv='fea.csv',num_classes=10)

    features = [row[:-1] for row in dataset]  # Extract features from each row
    labels = [row[-1] for row in dataset]  # Extract labels from each row

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)

    # Add the labels back to the dataset
    dataset_pca = [list(row) + [label] for row, label in zip(features_pca, labels)]

    # Write the dataset to a CSV file
    with open('pca.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(dataset_pca)

    avg_accuracy, avg_time = kNN_performance(csv='pca.csv')

    return avg_accuracy, avg_time, dataset_pca


def visualize_covariance_matrices(n_components):

    dataset = process_dataset(csv='fea.csv', num_classes=10)

    features = [row[:-1] for row in dataset]  # Extract features from each row
    labels = [row[-1] for row in dataset]  # Extract labels from each row

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)

    covariance_original = np.cov(features, rowvar=False)
    covariance_pca = np.cov(features_pca, rowvar=False)

    plt.subplot(1, 2, 1)
    plt.imshow(covariance_original, cmap='gnuplot', interpolation='nearest')
    plt.title('Covariance Matrix (Original)')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(covariance_pca, cmap='gnuplot', interpolation='nearest')
    plt.title('Covariance Matrix (PCA)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('covariance_matrices.png')
    plt.show()




"""---------------------------------------------------------------MAIN FUNCTION---------------------------------------------------------------"""

if __name__ == "__main__":

    knn_accuracy, knn_time = kNN_performance()
    print(f'KNN\nAccuracy: {knn_accuracy}   Time: {knn_time}')

    pca_accuracy, pca_time, dataset_pca = pca_performance(20)
    print(f'PCA\nAccuracy: {pca_accuracy}   Time: {pca_time}')

    visualize_covariance_matrices(20)
 