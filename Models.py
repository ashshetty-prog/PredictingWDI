import numpy as np
import pandas as pd
from collections import Counter

class WeightRelation:
    def __init__(self, train_features_set, validation_features_set, test_features_set):
        n_features = len(train_features_set[0])
        self.weights = [np.random.random() for _ in range(n_features)]

    @staticmethod
    def get_non_empty_features_indices(features):
        non_empty_features = []
        for i, f in enumerate(features):
            if f != "":
                non_empty_features.append(i)


if __name__ == '__main__':
    df = pd.read_csv("data/reduced_wdi_historical.csv")
    nulls = df.isnull().sum(axis=1).tolist()
    count = Counter()
    for e in nulls:
        count[e] += 1
    print(sorted(count.items()))
