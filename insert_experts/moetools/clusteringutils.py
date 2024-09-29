import time
import torch
import torch.nn as nn
from torch import pca_lowrank
from cuml import KMeans as cuml_KMeans
from kmeans_pytorch import kmeans_predict as pt_kmeans_predict
from moetools.moe_linear import ClusteredLinear

class torchPCA(nn.Module):

    def __init__(self,  hidden_dim, n_components, n_iter=500, center=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.n_iter = n_iter
        self.center = center
        
        V = torch.zeros((hidden_dim, n_components), dtype=torch.float32)
        mean = torch.zeros((1, hidden_dim), dtype=torch.float32)

        self.register_buffer("V", V)
        self.register_buffer("mean", mean)

    def _svd_flip_vector(self, U, V):

        max_abs_cols = torch.argmax(V.abs(), dim=-2)
        sign_cols = torch.sign(V[max_abs_cols, range(V.shape[-1])])
        V *= sign_cols
        U *= sign_cols
        return U, V
        
    def fit(self, X):
        data_type, dev = X.dtype, X.device

        self.mean = X.mean(dim=(-2,), keepdim=True)
        U, _, V = torch.pca_lowrank(X.float(), 
                                    q=self.n_components, 
                                    center=self.center, 
                                    niter=self.n_iter)
        _, self.V = self._svd_flip_vector(U, V)
        self.V = V
        self.V = self.V.to(dtype=data_type, device=dev)

    def forward(self, X):
        
        if (self.V is not None) and (self.mean is not None):
            Out = (X - self.mean) @ self.V
            return Out
        
        else:
            raise ValueError("self.V or self.mean is None, try to use `fit` method")
        

class torchKMeans(nn.Module):

    def __init__(self, hidden_dim, n_clusters, n_init=100, random_state=0):
        super().__init__()

        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

        self.labels = None
        cluster_centers = torch.zeros((n_clusters, hidden_dim), dtype=torch.float32)
        self.register_buffer("cluster_centers", cluster_centers)

    def fit(self, X):
        data_type, dev = X.dtype, X.device

        kmeans_model = cuml_KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        kmeans_model.fit(X.float())
        self.cluster_centers = torch.as_tensor(kmeans_model.cluster_centers_)
        self.cluster_centers = self.cluster_centers.to(dtype=data_type, device=dev)
        
        self.labels = torch.as_tensor(kmeans_model.labels_)
        self.labels = self.labels.to(dtype=torch.long, device=dev)

        return self.cluster_centers, self.labels

    def _pairwise_distance(self, data1, data2):
        # The function is provided by `kmeans_pytorch`
        
        # transfer to device
        data1, data2 = data1, data2

        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)

        dis = (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)

        return dis

    def _kmeans_predict(self, X):   
        dis = self._pairwise_distance(X, self.cluster_centers)
        choice_cluster = torch.argmin(dis, dim=1)
        
        return choice_cluster

    def forward(self, X):
        if (self.cluster_centers is not None):
            choice_cluster = self._kmeans_predict(X)
            # choice_cluster = pt_kmeans_predict(X, self.cluster_centers)
            return choice_cluster
        
        else:
            raise ValueError("self.cluster_centers is None, try using `fit` method")


def make_and_apply_PCA(inputs, hidden_dim, reduction_factor, verbose=False):
    if verbose:
        print(
            "Making and applying PCA with reduction factor: ", reduction_factor
        )
        tick = time.time()
    pca_model = torchPCA(
        hidden_dim=hidden_dim,
        n_components=hidden_dim // reduction_factor
    )
    pca_model.fit(inputs.reshape(-1, inputs.shape[-1]))
    transformed_inputs = pca_model(inputs.reshape(-1, inputs.shape[-1]))
    if verbose:
        tock = time.time()
        print("PCA model acquired")
        print("Time taken: ", tock - tick)

    return pca_model, transformed_inputs

def make_and_apply_KMeans(inputs, n_clusters, verbose=False):
    if verbose:
        print("Making and applying KMeans with n_clusters: ", n_clusters)
        tick = time.time()

    hidden_dim = inputs.shape[-1]
    kmeans_model = torchKMeans(hidden_dim=hidden_dim, n_clusters=n_clusters)
    _, labels = kmeans_model.fit(inputs.reshape(-1, inputs.shape[-1]))
    if verbose:
        tock = time.time()
        print("KMeans model acquired")
        print("Time taken: ", tock - tick)
        unique, counts =torch.unique(torch.as_tensor(labels), return_counts=True)
        total = counts.sum()
        print(f"Proportion of each cluster:")
        for i in range(len(unique)):
            print(f"{unique[i]}: {counts[i]/total}")

    return kmeans_model


def make_clustered(module, names, name="", num_clusters=1):
    if isinstance(module, ClusteredLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            orig_layer = getattr(module, attr)
            layer = ClusteredLinear(
                orig_layer.in_features, orig_layer.out_features,
                num_clusters=num_clusters
            )
            layer.weight = orig_layer.weight
            setattr(module, attr, layer)

    for name1, child in module.named_children():
        make_clustered(
            child,
            names,
            name + "." + name1 if name != "" else name1,
            num_clusters=num_clusters,
        )