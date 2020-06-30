# This program applies PCA to the MNIST database to speed up
# the fitting of machine learning algorithms.

#===========================================================
# Author: Marjan Khamesian
# Date: June 2020
#===========================================================

# Load the Data

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
#print(mnist)

# Images
mnist.data.shape

# Labels
mnist.target.shape

# === Splitting Data ===
from sklearn.model_selection import train_test_split

# the proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( 
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)

# Shape of data
print('The shape of training image is', train_img.shape)
print('The shape of training label is', train_lbl.shape)
print('The shape of test image is', test_img.shape)
print('The shape of test label is', test_lbl.shape)

# === Standardizing the Data ===

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# === Fitting === 
scaler.fit(train_img)

# === transformation ===
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# === PCA ===
from sklearn.decomposition import PCA

# Make an instance of the Model
pca = PCA(.95)

# Fitting PCA on training set
pca.fit(train_img)

# Number of components PCA choose after fitting the model
print('Number of components after fitting is:')
print(pca.n_components_)

# Transformation
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# === Apply Logistic Regression to the Transformed Data ===



