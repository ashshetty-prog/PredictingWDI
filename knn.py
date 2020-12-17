from collections import Counter
from sklearn.metrics.pairwise import nan_euclidean_distances


class KNN:
    def __init__(self, X, k):
        self.X_train = X
        self.k = k

    def findKNeighbors(self):
        corr_features = dict()
        for col1 in self.X_train.columns:
            d = []
            votes = []
            for col2 in self.X_train.columns:
                dist = nan_euclidean_distances(self.X_train[col1].values.reshape(1, -1),
                                               self.X_train[col2].values.reshape(1, -1))
                d.append([dist, col2])
            # sorts and picks the nearest one
            # print('neighbors', d)
            d.sort(key=lambda x: x[0])
            d = d[0:self.k]
            for v, j in d:
                votes.append(j)
            corr_features[col1] = votes
            # ans = Counter(votes).most_common(1)[0][0]
            # result.append(ans)

        return corr_features
