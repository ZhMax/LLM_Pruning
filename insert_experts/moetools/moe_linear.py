import numpy as np
import cupy as cp
import torch
import torch.nn as nn
from torch.nn import functional as F

from collections import OrderedDict

class ClusteredLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            num_clusters=1, 
            perc=0.0
        ):

        super().__init__(in_features, out_features, bias=False)
        self.perc = perc
        
        # Three dimensional tensor of pruning masks
        pruning_masks = None
        self.register_buffer("pruning_masks", pruning_masks)

        self.freqs = np.array([0])
        self.mode = "calibrate"

        self.kmeans_model = None
        self.pca_model = None
        self.num_clusters = num_clusters
        self.batch_counter = 0

    def add_mask(self, mask):
        if self.pruning_masks is None:
            self.pruning_masks = mask.unsqueeze(0)
        else:
            self.pruning_masks = torch.stack(
                [self.pruning_masks, mask.unsqueeze(0)], 
                dim=0
            )
        self.freqs = np.append(self.freqs, 0)

    def reset_counter(self):
        self.batch_counter = 0

    def forward(self, X):
        nobatch = len(X.shape) == 2
        if nobatch:
            X = X.unsqueeze(0)

        if self.mode == "calibrate":
            Y = F.linear(X, self.weight)
            if not nobatch:
                Y = Y.unsqueeze(0)
            self.batch_counter += 1
            return Y

        batch_size = X.shape[0]
        sequence_length = X.shape[1]
        hidden_dimension = X.shape[2]

        if self.mode == "train":
            interm = self.kmeans_model.labels
            cluster_assignment = interm.reshape(-1, sequence_length)[
                self.batch_counter
            ]

        elif self.mode == "test":
            if self.pca_model is not None:
                X_reduced = self.pca_model(X.reshape(-1, hidden_dimension))
                cluster_assignment = self.kmeans_model(X_reduced)
                cluster_assignment = cluster_assignment.reshape(sequence_length,)

            else:
                X_reduced = self.pca_model(X.reshape(-1, hidden_dimension))
                cluster_assignment = self.kmeans_model(X_reduced)
                cluster_assignment = cluster_assignment.reshape(sequence_length,)

        
        Y_final = torch.zeros((sequence_length, self.out_features), 
                              device=self.weight.device, dtype=self.weight.dtype)
        # Y = self.linear(X)
        # Y_final = torch.zeros_like(Y[0])
        for i in range(self.num_clusters):
            idx_cluster = cluster_assignment == i

            X_cluster = X[:, idx_cluster, :]
            # Y = self.layers[i + 1](X_cluster)
            sparse_weight = self.pruning_masks[i].squeeze(0) * self.weight
            Y = F.linear(X_cluster, sparse_weight)
            Y_final[idx_cluster] = Y[0]

        self.batch_counter += 1

        if not nobatch:
            Y_final = Y_final.unsqueeze(0)

        del Y
        torch.cuda.empty_cache()
        return Y_final

"""
class ClusteredLinear(nn.Module):
    def __init__(self, linear=None, num_clusters=1, perc=0.0):
        super().__init__()
        self.perc = perc
        self.layers = nn.ModuleList()
        if linear is not None:
            self.layers.append(linear)
        self.freqs = np.array([0])
        self.mode = "calibrate"

        self.kmeans_model = None
        self.pca_model = None
        self.num_clusters = num_clusters
        self.batch_counter = 0

    def add_layer(self, layer):
        self.layers.append(layer)
        self.freqs = np.append(self.freqs, 0)

    def reset_counter(self):
        self.batch_counter = 0

    def forward(self, X):
        nobatch = len(X.shape) == 2
        if nobatch:
            X = X.unsqueeze(0)

        if self.mode == "calibrate":
            Y = self.layers[0](X)
            if not nobatch:
                Y = Y.unsqueeze(0)
            self.batch_counter += 1
            return Y

        batch_size = X.shape[0]
        sequence_length = X.shape[1]
        hidden_dimension = X.shape[2]

        if self.mode == "train":
            interm = self.kmeans_model.labels
            cluster_assignment = interm.reshape(-1, sequence_length)[
                self.batch_counter
            ]

        elif self.mode == "test":
            if self.pca_model is not None:
                X_reduced = self.pca_model(X.reshape(-1, hidden_dimension))
                cluster_assignment = self.kmeans_model(X_reduced)
                cluster_assignment = cluster_assignment.reshape(sequence_length,)

            else:
                X_reduced = self.pca_model(X.reshape(-1, hidden_dimension))
                cluster_assignment = self.kmeans_model(X_reduced)
                cluster_assignment = cluster_assignment.reshape(sequence_length,)

        
        # Y_final = torch.zeros((sequence_length, self.layers[-1].out_features), 
        #                       device=X.device, dtype=X.dtype)
        Y = self.layers[-1](X)
        Y_final = torch.zeros_like(Y[0])
        for i in range(self.num_clusters):
            idx_cluster = cluster_assignment == i

            X_cluster = X[:, idx_cluster, :]
            Y = self.layers[i + 1](X_cluster)
            Y_final[idx_cluster] = Y[0]

        self.batch_counter += 1

        if not nobatch:
            Y_final = Y_final.unsqueeze(0)

        del Y
        torch.cuda.empty_cache()
        return Y_final
"""