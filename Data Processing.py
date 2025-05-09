# EE 460 Final Project
# Group 1
# Heart Disease Predictor

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

#### CHANGE THIS TO YOUR DATASET LOCATION ##############
input_file = './project/heart_disease_uci.csv'
########################################################

def read_data(data_location):
    data = pd.read_csv(data_location, header=None)
    # data = shuffle(data, random_state=42)
    data = data.to_numpy()
    rows, cols = data.shape
    Class = data[1:,15]
    data = data[1:,0:15]
    return data, Class, rows, cols

data,Class,rows,cols= read_data(input_file)
# Preprocessing ########################################
# Remove "slope", "ca", "thal" features
data = data[:,0:12]

# Change FALSE/TRUE to 0/1 and Female/Male to 0/1
data[:, 2] = np.where(data[:, 2] == 'Male', 1, 0)
data[:, 7] = np.where(data[:, 7] == 'TRUE', 1, 0)
data[:, 10] = np.where(data[:, 10] == 'TRUE', 1, 0)

# Fill in missing feature data
def average_replace(dataset, column):
    col = data[:, column].astype(float)
    data[np.isnan(col), column] = np.nanmean(col)
    return dataset

# Replace missing "trestps" with the average
data = average_replace(data,5)

# Replace missing "chol" with the average
data = average_replace(data,6)

# Replace missing "fbs" with the mode
# mode is FALSE, and was taken care of in line 39
# TRUE is replaced with 1, everything else is 0

# Replace missing "restecg" with the mode
data[pd.isna(data[:,8]), 8] = "normal"

# Replace missing "thalch" with the average
data = average_replace(data,9)

# Replace missing "exang" with the mode
# mode is FALSE, and was taken care of in line 40
# TRUE is replaced with 1, everything else is 0

# Replace missing "oldpeak" with the average
data = average_replace(data,11)

# One-hot encoding for categorical data ("cp", "restecg" features)
categorical_cols = data[:, [4, 8]]
categorical_cols = categorical_cols.reshape(-1, len([4, 8]))
encoder = OneHotEncoder(sparse=False, dtype=int)
one_hot = encoder.fit_transform(categorical_cols)
data = np.delete(data, [4, 8], axis=1) # Delete original features
data = np.hstack((data, one_hot)) # Add one-hot features
# print(encoder.categories_)

# Normalize numerical data ("age", "trestbps", "chol", "thalch", "oldpeak")
cols_to_normalize = [1, 4, 5, 7, 9] # Indexes are different b/c categorical data was concatenated to end
features_to_scale = data[:, cols_to_normalize]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_to_scale)
for i, col in enumerate(cols_to_normalize):
    data[:, col] = scaled_features[:, i]

# Drop index column, and dataset column
data = data[:,1:]
data = np.delete(data, 2, axis=1)

# Upsampling using SMOTE
categorical_features = [1] + [4] + [6] + list(range(8, 15))

# Initialize SMOTENC with the categorical features
smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)

# Apply SMOTE to upsample the dataset
upsampled_X, upsampled_y = smote_nc.fit_resample(data, Class)

# Check for class balance
# unique_labels, counts = np.unique(upsampled_y, return_counts=True)
# label_counts = dict(zip(unique_labels, counts))
# print("Unique labels and their counts:", label_counts)

def dataset_split(dataset, classes):
    # Splits into 80/10/10 (train/val/test) with class balance
    train_val_data, test_data, train_val_class, test_class = train_test_split(
    dataset, classes, test_size=0.1, random_state=42, stratify=classes)
    
    train_data, val_data, train_class, val_class = train_test_split(
    train_val_data, train_val_class, test_size=0.11, random_state=42, stratify=train_val_class)
    return train_data, train_class, val_data, val_class, test_data, test_class

## data, Class is processed data without upsampling
## upsampled_X, upsampled_y is processed data with upsampling

## Train/Val/Test split datasets without upsampling
train_data, train_class, val_data, val_class, test_data, test_class = dataset_split(data, Class)
print(train_data.shape)
print(train_class.shape)
print(val_data.shape)
print(val_class.shape)
print(test_data.shape)
print(test_class.shape)

## Train/Val/Test split datasets with upsampling
Utrain_data, Utrain_class, Uval_data, Uval_class, Utest_data, Utest_class = dataset_split(upsampled_X, upsampled_y)

print(Utrain_data.shape)
print(Utrain_class.shape)
print(Uval_data.shape)
print(Uval_class.shape)
print(Utest_data.shape)
print(Utest_class.shape)