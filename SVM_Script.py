import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

image_paths = {
    'cat': 'images\cat',
    'couch': 'images\couch',
    'person': 'images\person'
}

# Epochs and initial clusters
epochs = 10
initial_clusters = 50  

sift = cv2.SIFT_create()

def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# Collect all descriptors
all_descriptors = []
image_labels = []
for label, folder in image_paths.items():
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        descriptors = extract_sift_features(img)
        if descriptors is not None:
            all_descriptors.append(descriptors)
            image_labels.append(label)

# Stack all descriptors vertically in a numpy array
all_descriptors = np.vstack(all_descriptors)

best_accuracy = 0.0
best_model = None

for epoch in range(epochs):
    num_clusters = initial_clusters + epoch * 10  
    print(f"Epoch {epoch + 1}/{epochs} with {num_clusters} clusters")

    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    visual_words = kmeans.cluster_centers_

    def create_feature_vector(descriptors, kmeans):
        visual_words_hist = np.zeros(num_clusters)
        if descriptors is not None:
            clusters = kmeans.predict(descriptors)
            for c in clusters:
                visual_words_hist[c] += 1
        return visual_words_hist

    global_features = []
    for label, folder in image_paths.items():
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            descriptors = extract_sift_features(img)
            if descriptors is not None:
                feature_vector = create_feature_vector(descriptors, kmeans)
                global_features.append(feature_vector)

    global_features = np.array(global_features)
    image_labels = np.array(image_labels)

    label_mapping = {'cat': 0, 'couch': 1, 'person': 2}
    y = np.array([label_mapping[label] for label in image_labels])

    X_train, X_test, y_train, y_test = train_test_split(global_features, y, test_size=0.3, random_state=42)

    # Train SVM classifier
    svm = SVC(kernel='linear')  # You can change the kernel to 'rbf', 'poly', etc. based on experimentation
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Epoch {epoch + 1} accuracy: {accuracy * 100:.2f}%")

    # Check if this is the best model so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = svm
        print(f"New best model found at epoch {epoch + 1} with accuracy: {best_accuracy * 100:.2f}%")

# Save the best model
with open('best_svm_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

print(f"Best model accuracy: {best_accuracy * 100:.2f}%")
