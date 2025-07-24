import numpy as np
from collections import Counter
import sys
import os

# Ensure MNIST_Loader can be imported
sys.path.append(os.path.abspath("../MNIST_Dataset_Loader"))
from mnist_loader import MNIST


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def main():
    print("📥 Loading MNIST data...")
    loader = MNIST(path='../MNIST_Dataset_Loader/dataset')

    X_train_raw, y_train_raw = loader.load_training()
    X_test_raw, y_test_raw = loader.load_testing()

    # Normalize and convert to numpy
    X_train = np.array(X_train_raw[:1000]) / 255.0
    y_train = np.array(y_train_raw[:1000])
    X_test = np.array(X_test_raw[:100]) / 255.0
    y_test = np.array(y_test_raw[:100])

    print("🔢 Training KNN classifier...")
    clf = KNN(k=3)
    clf.fit(X_train, y_train)

    print("🧪 Evaluating...")
    predictions = clf.predict(X_test)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"✅ Accuracy on 100 test samples: {accuracy * 100:.2f}%")

    # Display one example
    print(f"🔍 Predicted: {predictions[0]}, Actual: {y_test[0]}")
    print(loader.display(X_test_raw[0]))


if __name__ == "__main__":
    main()
