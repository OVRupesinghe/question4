import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

image_paths = {
    'cat': 'images\cat',
    'couch': 'images\couch',
    'person': 'images\person'
}

sift = cv2.SIFT_create()

def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def compute_statistics(descriptors):
    if descriptors is not None:
        mean = np.mean(descriptors, axis=0)
        variance = np.var(descriptors, axis=0)
        minimum = np.min(descriptors, axis=0)
        maximum = np.max(descriptors, axis=0)
        global_vector = np.concatenate([mean, variance, minimum, maximum])
        return global_vector
    else:
        return np.zeros(4 * 128)

global_features = []
image_labels = []

for label, folder in image_paths.items():
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        descriptors = extract_sift_features(img)
        global_vector = compute_statistics(descriptors)
        global_features.append(global_vector)
        image_labels.append(label)

global_features = np.array(global_features)
image_labels = np.array(image_labels)

label_mapping = {'cat': 0, 'couch': 1, 'person': 2}
y = np.array([label_mapping[label] for label in image_labels])

X_train, X_test, y_train, y_test = train_test_split(global_features, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"kNN Classifier Accuracy: {knn_accuracy * 100:.2f}%")

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Classifier Accuracy: {svm_accuracy * 100:.2f}%")


if knn_accuracy > svm_accuracy:
    best_model = knn
    with open('best_knn_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    print("Best model: kNN")
else:
    best_model = svm
    with open('best_svm_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    print("Best model: SVM")