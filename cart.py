import numpy as np
import pandas as pd


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


def calculate_gini(y):
    classes = np.unique(y)
    num_samples = len(y)
    gini = 1.0

    for c in classes:
        p = np.sum(y == c) / num_samples
        gini -= p ** 2

    return gini


def split_dataset(X, y, feature_index, threshold):
    mask = X[:, feature_index] <= threshold
    left_X, left_y = X[mask], y[mask]
    right_X, right_y = X[~mask], y[~mask]
    return left_X, left_y, right_X, right_y


def find_best_split(X, y):
    m, n = X.shape
    if m <= 1:
        return None, None

    num_parent = [np.sum(y == c) for c in np.unique(y)]
    best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
    best_idx, best_thr = None, None

    for idx in range(n):
        thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
        num_left = [0] * len(np.unique(y))
        num_right = num_parent.copy()

        for i in range(1, m):
            c = classes[i - 1]
            num_left[c] += 1
            num_right[c] -= 1
            gini_left = 1.0 - sum((num_left[x] / (i + 1)) ** 2 for x in np.unique(y))
            gini_right = 1.0 - sum((num_right[x] / (m - i + 1)) ** 2 for x in np.unique(y))
            gini = (i * gini_left + (m - i) * gini_right) / m
            if thresholds[i] == thresholds[i - 1]:
                continue
            if gini < best_gini:
                best_gini = gini
                best_idx = idx
                best_thr = (thresholds[i] + thresholds[i - 1]) / 2

    return best_idx, best_thr


def create_tree(X, y):
    num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
    predicted_class = np.argmax(num_samples_per_class)
    node = Node(
        gini=calculate_gini(y),
        num_samples=len(y),
        num_samples_per_class=num_samples_per_class,
        predicted_class=predicted_class,
    )

    idx, thr = find_best_split(X, y)
    if idx is None:
        return node

    left_X, left_y, right_X, right_y = split_dataset(X, y, idx, thr)
    node.feature_index = idx
    node.threshold = thr
    node.left = create_tree(left_X, left_y)
    node.right = create_tree(right_X, right_y)
    return node


def predict_tree(node, X):
    if node.left is None and node.right is None:
        return node.predicted_class

    if X[node.feature_index] <= node.threshold:
        return predict_tree(node.left, X)
    else:
        return predict_tree(node.right, X)


# Przykład użycia dla klasyfikacji
def main_classification():
    # Wczytanie danych
    dataset = pd.read_csv("titanic.csv")
    dataset = dataset[["Age", "Sex", "Survived"]]
    dataset = dataset.replace("male", 0)
    dataset = dataset.replace("female", 1)
    print(dataset.head())
    X = dataset.drop('Survived', axis=1).values
    y = dataset['Survived'].values

    # Utworzenie drzewa decyzyjnego
    tree = create_tree(X, y)

    # Przykładowa predykcja
    sample = X[0]
    prediction = predict_tree(tree, sample)
    print(f"Przykładowa predykcja: {prediction}")


if __name__ == "__main__":
    main_classification()
