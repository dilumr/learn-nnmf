import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


class ClusterModel:

    @property
    def clusters(self):
        return len(self._cluster_names)

    @property
    def cluster_names(self):
        return self._cluster_names

    @clusters.setter
    def clusters(self, value):
        self._cluster_names = [f'C_{i}' for i in range(0, value)]

    @property
    def ratings_matrix(self):
        return self._X

    @ratings_matrix.setter
    def ratings_matrix(self, ratings_dfm):
        self._X = ratings_dfm  # 'V' in some literature

    def build(self):
        assert self.ratings_matrix is not None
        assert self.cluster_names is not None
        model = NMF(n_components=self.clusters, init='random', random_state=0)

        self._W  = model.fit_transform(self._X)  # 'features' matrix
        self._H  = model.components_             # 'coefficients' matrix
        self._err = model.reconstruction_err_    # divergence between W.H and X

        return self._err

    def reconstruct(self, round_decimals=None):
        assert self._X is not None
        assert self._W is not None
        assert self._H is not None
        Xhat = self._W.dot(self._H)
        if round_decimals and round_decimals > 0:
            Xhat = np.round(Xhat, decimals=round_decimals)
        return pd.DataFrame(Xhat, index=self._X.index, columns=self._X.columns)

    @property
    def subject_cluster_dfm(self):
        return pd.DataFrame(self._W, index=self._X.index, columns=self._cluster_names)

    @property
    def object_cluster_dfm(self):
        # Note intentional transport to orient H consistent with W.
        return pd.DataFrame(self._H, index=self._cluster_names, columns=self._X.columns).T
