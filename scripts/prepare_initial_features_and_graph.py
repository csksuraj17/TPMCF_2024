import pandas as pd
import numpy as np
import copy
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
import networkx as nx
import scipy.sparse as sp
import warnings
import matplotlib.pyplot as plt
from scipy import sparse
import os

def load_sparse_matrix(filename_prefix, time_index):
    return sparse.load_npz(f"{filename_prefix}_t{time_index}.npz")

def similarity_feature(csm_data):
    csm_user = cosine_similarity(csm_data)
    csm_serv = cosine_similarity(csm_data.T)
    return csm_user, csm_serv

# def euclidean_dist(euclid_data):
#     u_euclid = euclidean_distances(euclid_data, euclid_data)
#     s_euclid = euclidean_distances(euclid_data.T, euclid_data.T)
#     for mat in [u_euclid, s_euclid]:
#         for i in range(mat.shape[0]):
#             s = mat[i].sum()
#             mat[i] = mat[i] / s if s != 0 else mat[i]
#     return u_euclid, s_euclid

def svd_feature(svd_data):
    U_svd, sigma, S_svd = svds(svd_data, k=50)
    return U_svd, S_svd.T

def NormalizeData(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def statistical_feature(data_df):
    data_df = np.array(pd.DataFrame(data_df).replace(0, np.nan))
    stats = lambda axis: [NormalizeData(func(data_df, axis=axis)) for func in [np.nanmin, np.nanmax, np.nanmedian, np.nanmean, np.nanstd]]
    user_stat = np.array(stats(axis=1)).T
    serv_stat = np.array(stats(axis=0)).T
    user_stat = np.nan_to_num(user_stat)
    serv_stat = np.nan_to_num(serv_stat)
    return user_stat, serv_stat

def plot_loss(model):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

class User_AE(Model):
    def __init__(self, output_units):
        super().__init__()
        self.encoder = Sequential([
            Dense(120, activation="tanh"), Dropout(0.4),
            Dense(80, activation="tanh"), Dropout(0.4),
            Dense(50, activation="linear")
        ])
        self.decoder = Sequential([
            Dense(80, activation="tanh"), Dropout(0.4),
            Dense(120, activation="tanh"), Dense(output_units, activation="linear")
        ])
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

def user_train_and_test(feature_data):
    model = User_AE(feature_data.shape[1])
    model.compile(loss='mse', metrics=['mae'], optimizer='adam')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    model.fit(feature_data, feature_data, verbose=0, epochs=500, callbacks=[callback], validation_split=0.2)
    return model.encoder.predict(feature_data)

class Serv_AE(Model):
    def __init__(self, output_units):
        super().__init__()
        self.encoder = Sequential([
            Dense(1024, activation="tanh"), Dropout(0.4),
            Dense(256, activation="tanh"), Dropout(0.4),
            Dense(50, activation="linear")
        ])
        self.decoder = Sequential([
            Dense(256, activation="tanh"), Dropout(0.4),
            Dense(1024, activation="tanh"), Dense(output_units, activation="linear")
        ])
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

def serv_train_and_test(feature_data):
    model = Serv_AE(feature_data.shape[1])
    model.compile(loss='mse', metrics=['mae'], optimizer='adam')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    model.fit(feature_data, feature_data, verbose=0, epochs=500, callbacks=[callback], validation_split=0.2)
    return model.encoder.predict(feature_data)

def degree_power(A, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)).ravel(), k)
    degrees[np.isinf(degrees)] = 0.0
    return sp.diags(degrees) if sp.issparse(A) else np.diag(degrees)

def normalized_adjacency(adj_indicator):
    D = degree_power(adj_indicator, -0.5)
    norm_adj = D @ adj_indicator @ D
    return np.identity(adj_indicator.shape[0]) + norm_adj

def adj_calculation(temp):
    G = nx.Graph()
    edge_list = [(f"U_{i}", f"S_{j}", temp[i][j]) for i in range(temp.shape[0]) for j in range(temp.shape[1]) if temp[i][j] != 0]
    G.add_weighted_edges_from(edge_list)
    adj_mat = nx.to_numpy_array(G)
    adj_indicator = (adj_mat != 0).astype(float)
    return normalized_adjacency(adj_indicator)

def run_feature_generation():
    # === Feature Generation for Sparse Matrices ===
    datasets = ['rt', 'tp']
    percents = [5, 10, 15, 20]
    prev_time_steps = 8
    max_time_steps = 64
    sparse_path = './sparse_files/{}'  # Sparse .npz files like rt_5_t63.npz
    output_path = './data_files/{}'
    
    with tf.device('gpu'):
        for dataset in datasets:
            for percent in percents:
                print(f"Working for {dataset}-{percent}%")
                handcrafted_features, adj_norms = [], []
                
                for k in range(max_time_steps-prev_time_steps, max_time_steps):  # last prev_time_steps
                    temp_sparse = load_sparse_matrix(sparse_path.format(f'{dataset}_{percent}'), k)
                    temp = temp_sparse.toarray()
    
                    csm_user, csm_serv = similarity_feature(temp)
                    csm_user = user_train_and_test(csm_user)
                    csm_serv = serv_train_and_test(csm_serv)
    
                    user_stat, serv_stat = statistical_feature(temp)
                    user_svd, serv_svd = svd_feature(temp)
    
                    csm_feature = np.concatenate((csm_user, csm_serv), axis=0)
                    svd_feat = np.concatenate((user_svd, serv_svd), axis=0)
                    stat_feature = np.concatenate((user_stat, serv_stat), axis=0)
    
                    node_feature = np.concatenate((csm_feature, svd_feat, stat_feature), axis=1)
                    handcrafted_features.append(node_feature)
    
                    adj_norm = adj_calculation(temp)
                    adj_norms.append(adj_norm)
    
                np.save(output_path.format(f'{dataset}_{percent}_handcrafted_features.npy'), handcrafted_features)
                np.save(output_path.format(f'{dataset}_{percent}_adj_norm.npy'), adj_norms)
