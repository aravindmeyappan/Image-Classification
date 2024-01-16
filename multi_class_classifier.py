import itertools
from sklearn.svm import SVC
import numpy as np

class OneVsAllSVM:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.models = []

    def fit(self, X, y):
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        for i in range(num_classes):
            # Create binary labels for the current class
            binary_labels = np.where(y == unique_classes[i], 1, 0)

            # Train a binary SVM for the current class
            model = SVC(kernel=self.kernel, C=self.C)
            model.fit(X, binary_labels)
            self.models.append(model)

    def predict(self, X):   
        # Make predictions using all binary classifiers
        predictions = [model.predict(X) for model in self.models]

        # Select the class with the highest decision value
        predicted_labels = np.argmax(predictions, axis=0)

        # Map back to original class labels
        unique_classes = np.unique(predicted_labels)
        class_mapping = {i: unique_classes[i] for i in range(len(unique_classes))}
        mapped_labels = np.vectorize(class_mapping.get)(predicted_labels)

        return mapped_labels
    

class OneVsOneSVM:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.models = []

    def fit(self, X, y):
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        # Create all possible pairs of classes
        class_pairs = list(itertools.combinations(unique_classes, 2))

        for pair in class_pairs:
            # Select data points corresponding to the current pair of classes
            pair_mask = np.logical_or(y == pair[0], y == pair[1])
            X_pair = X[pair_mask]
            y_pair = y[pair_mask]

            # Convert labels to -1 and 1 for SVM
            y_pair_svm = np.where(y_pair == pair[0], 1, -1)

            # Train a binary SVM for the current pair of classes
            model = SVC(kernel=self.kernel, C=self.C)
            model.fit(X_pair, y_pair_svm)
            self.models.append((pair, model))

    def predict(self, X):
        # Initialize an array to store the decision values for each pair
        decision_values = np.zeros((X.shape[0], len(self.models)))

        for i, (pair, model) in enumerate(self.models):
            # Make predictions using the binary classifier for the current pair
            decision_values[:, i] = model.predict(X)

        # Select the class with the most votes (highest cumulative decision value)
        predicted_labels = np.argmax(np.sum(decision_values, axis=1), axis=0)

        # Map back to original class labels
        unique_classes = np.unique([pair[0] for pair, _ in self.models] + [pair[1] for pair, _ in self.models])
        class_mapping = {i: unique_classes[i] for i in range(len(unique_classes))}
        mapped_label = class_mapping[predicted_labels]

        return mapped_label