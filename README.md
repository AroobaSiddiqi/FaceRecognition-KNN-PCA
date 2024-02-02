# FaceRecognition-KNN-PCA
This repository contains the implementation and analysis of face recognition techniques using K Nearest Neighbours (k-NN) and Principal Component Analysis (PCA) on a subset of the CMU Pose, Illumination, and Expression (PIE) database.

# KNN Classifier with PCA

This repository contains Python code implementing a K-nearest neighbors (KNN) classifier along with Principal Component Analysis (PCA) for dimensionality reduction. The KNN algorithm is a simple and effective method for classification tasks, while PCA is a technique commonly used for reducing the dimensionality of large datasets.

# FaceRecognition-KNN-PCA
This repository contains the implementation and analysis of face recognition techniques using K Nearest Neighbours (k-NN) and Principal Component Analysis (PCA) on a subset of the CMU Pose, Illumination, and Expression (PIE) database.

## Libraries Used
- numpy: Fundamental package for scientific computing with Python.
- csv: Library to read and write CSV files.
- random: Library for generating random numbers and shuffling sequences.
- time: Module for measuring time-related functions.
- matplotlib: Comprehensive library for creating static, animated, and interactive visualizations in Python.
- pandas: Data manipulation and analysis library.
- sklearn.decomposition.PCA: Module for performing Principal Component Analysis (PCA).
- scipy.spatial.distance: Module for computing distances between objects.

## Contents

1. [Dataset Processing](#dataset-processing)
2. [KNN Classifier](#knn-classifier)
3. [PCA](#pca)
4. [Performance Evaluation](#performance-evaluation)
5. [Main Function](#main-function)

## Dataset Processing

The `read_csv`, `normalize_dataset`, `label_dataset`, and `train_test_split` functions are provided to preprocess the dataset. These functions read the CSV file, normalize the data, add labels, and split the dataset into training and testing sets.

## KNN Classifier

The KNN classifier implementation consists of functions for calculating distances (Euclidean, Mahalanobis, and cosine similarity), finding nearest neighbors, and performing majority voting to make predictions. The `kNN_classifier` function orchestrates the entire classification process.

## PCA

Principal Component Analysis (PCA) is performed to reduce the dimensionality of the dataset. The `pca_performance` function calculates the accuracy and time taken for classification after PCA transformation. Additionally, `visualize_covariance_matrices` function is provided to visualize covariance matrices before and after PCA.

## Performance Evaluation

The `kNN_performance` function evaluates the performance of the KNN classifier by calculating accuracy and execution time. The function can be configured with parameters such as the number of nearest neighbors (k), distance metric, number of classes, and number of training instances.

## Main Function

The `main` function of the script demonstrates the usage of the implemented algorithms. It performs KNN classification and PCA transformation on the dataset, and prints out the accuracy and time taken for both methods. Additionally, it visualizes the covariance matrices before and after PCA.
