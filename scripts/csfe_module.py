import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer
import copy
from tqdm import tqdm
from utils import load_sparse_matrix, load_dense_matrix


class Custom_Loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        indicator = tf.cast(y_true > 0, tf.float32)
        return tf.reduce_mean(tf.math.log(1 + tf.square((y_pred - y_true) * indicator)))

class GConv(Layer):
    def __init__(self, adj, units=32, activation=tf.nn.relu):
        super(GConv, self).__init__()
        self.adj = tf.cast(adj, tf.float32)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(input_shape[-1], 128), initializer="random_normal", trainable=True)
        self.b1 = self.add_weight(shape=(128,), initializer="random_normal", trainable=True)
        self.w2 = self.add_weight(shape=(128, 64), initializer="random_normal", trainable=True)
        self.b2 = self.add_weight(shape=(64,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        x = tf.matmul(self.adj, tf.matmul(inputs, self.w1)) + self.b1
        x = self.activation(x)
        x = tf.matmul(x, self.w2) + self.b2
        return self.activation(x)

class GCNBlock(Layer):
    def __init__(self, adj, num_users, num_servs):
        super(GCNBlock, self).__init__()
        self.adj = adj
        self.conv1 = GConv(self.adj, activation=tf.nn.relu)
        self.conv2 = GConv(self.adj, activation=tf.nn.relu)
        self.NU = num_users
        self.NS = num_servs

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        all_embed = tf.concat([x1, x2], axis=-1)
        u_embed, s_embed = tf.split(all_embed, [NU, NS], axis=0)
        pred = tf.matmul(u_embed, s_embed, transpose_b=True)
        return pred, all_embed

def gcn_train_and_predict(adj, features, labels, num_users, num_servs):
    model_input = Input(shape=(features.shape[1],))
    gcn_block = GCNBlock(adj, num_users, num_servs)
    model = Model(inputs=model_input, outputs=gcn_block(model_input))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = Custom_Loss()
    best_loss = float('inf')
    wait = 0
    patience = 300

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            preds, _ = model(features, training=True)
            loss = loss_fn(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(10000):
        loss = train_step()
        if loss < best_loss:
            best_loss = loss
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    pred_rt, embedding = model(features, training=False)
    return embedding.numpy(), pred_rt.numpy()

def t_gcn_training(num_users, num_servs, num_time_steps, prev_time_steps, FEATURE_DIR, ADJ_DIR, DATA_DIR, OUT_DIR):
    total_time = 0
    for t in tqdm(range(prev_time_steps-prev_time_steps, num_time_steps)):
        features = np.load(os.path.join(FEATURE_DIR, f'features_t{t}.npy'))
        adj = load_sparse_matrix(os.path.join(ADJ_DIR, f'adj_t{t}.npz')).todense()
        label = load_dense_matrix(os.path.join(DATA_DIR, f'train_matrix_t{t}.npy'))

        features = tf.convert_to_tensor(features, dtype=tf.float32)
        labels = tf.convert_to_tensor(label, dtype=tf.float32)

        start = time.time()
        embedding, prediction = gcn_train_and_predict(adj, features, labels, num_users, num_servs)
        total_time += time.time() - start

        np.save(os.path.join(OUT_DIR, f'gcn_embed_t{t}.npy'), embedding)
        np.save(os.path.join(OUT_DIR, f'gcn_pred_t{t}.npy'), prediction)

    print(f"GCN training completed in {total_time:.2f} seconds.")
