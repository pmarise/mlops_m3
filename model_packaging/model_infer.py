import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 
import pickle
# Directories for the images
urban_path = 'urban'
rural_path = 'rural'

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to 128x128
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    equalized = cv2.equalizeHist(gray)  # Histogram equalization
    smoothed = cv2.GaussianBlur(equalized, (5, 5), 0)  # Gaussian smoothing
    return smoothed

# Load dataset
def load_dataset(urban_path, rural_path):
    data = []
    labels = []
    for file in os.listdir(urban_path):
        data.append(preprocess_image(os.path.join(urban_path, file)))
        labels.append(0)  # Urban = 0
    for file in os.listdir(rural_path):
        data.append(preprocess_image(os.path.join(rural_path, file)))
        labels.append(1)  # Rural = 1
    return np.array(data), np.array(labels)

# Load and preprocess images
data, labels = load_dataset(urban_path, rural_path)




# HOG Features
def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True )
    return features

# LBP Features
def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    return np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)[0]

# Edge Detection Features
def extract_edge_features(image):
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    return edges

# Extract features for a dataset
#
def extract_features(data,feature_type):
    features = []

    for image in data:
        #print(image)
        hog_features = extract_hog_features(image)
        lbp_features = extract_lbp_features(image)
        edge_features = extract_edge_features(image)
        hog_features_edge = extract_hog_features(edge_features)
        lbp_features_edge = extract_lbp_features(edge_features)
        edge_hog_lbp = extract_lbp_features(edge_features)
        if feature_type == 'hog':
            features.append(np.hstack([hog_features]))
        elif feature_type == 'lbp':
            features.append(np.hstack([lbp_features]))
        elif feature_type == 'edge':
            features.append(np.hstack([edge_features.flatten()]))
        elif feature_type == 'lbp_edge':
            features.append(np.hstack([lbp_features_edge]))
        elif feature_type == 'hog_edge':
            features.append(np.hstack([hog_features_edge]))
        elif feature_type == 'hog_lbp_with_edge':
            features.append(np.hstack([hog_features_edge,lbp_features_edge]))

    return np.array(features)



def run_inference():
    with open('mlopsM3_cvimg_classifier.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded from mlopsM3_cvimg_classifier.pkl")
    Filter='hog_lbp_with_edge'
    load_dataset(urban_path, rural_path)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train_features = extract_features(X_train,Filter)
    print(X_train_features.shape)
    X_test_features = extract_features(X_test,Filter)
    print('lock:',X_train_features.shape)
    # Normalize features
    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)
    rf_predictions = loaded_model.predict(X_test_features)
    print(rf_predictions)
    return rf_predictions

def main():
    run_inference() 
  
if __name__ == '__main__':  
    main()  
