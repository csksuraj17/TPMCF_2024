
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.notebook import tqdm as tqdm_notebook
import time
import copy
import os
from csfe_module import fill_matrix, cal_metric, plot_loss


def cauchy_loss_fn(y_true, y_pred):
    loss = tf.math.log(1+tf.square(y_pred - y_true))
    return loss

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = embed_dim // num_heads

        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.out_dense = layers.Dense(embed_dim)
        self.attn_dropout = layers.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        query = self.split_heads(self.query_dense(inputs), batch_size)
        key   = self.split_heads(self.key_dense(inputs), batch_size)
        value = self.split_heads(self.value_dense(inputs), batch_size)

        scores = tf.matmul(query, key, transpose_b=True)
        scaled_scores = scores / tf.math.sqrt(tf.cast(self.proj_dim, tf.float32))
        weights = tf.nn.softmax(scaled_scores, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        attention = tf.matmul(weights, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        return self.out_dense(concat_attention)

def transformer_encoder(inputs, mha_layer, ff_dim, dropout=0.0):
    attn_output = mha_layer(inputs)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    x = tf.transpose(out1, perm=[0, 2, 1])
    ffn = layers.Conv1D(filters=ff_dim, kernel_size=3, activation='relu', padding='same')(x)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Conv1D(filters=inputs.shape[-2], kernel_size=1)(ffn)
    ffn = tf.transpose(ffn, perm=[0, 2, 1])
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.0, mlp_dropout=0.0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    mha_layer = MultiHeadSelfAttention(embed_dim=head_size, num_heads=num_heads, dropout=dropout)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, mha_layer, ff_dim, dropout)

    x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


def transformer_inputs(gcn_feature, train_matrix):
    with tf.device('gpu'):
        user_f, serv_f = tf.split(gcn_feature, [142, 4500], 1)
        X_train, Y_train, X_pred = [], [], []

        for i in tqdm_notebook(range(train_matrix.shape[0])):
            for j in range(train_matrix.shape[1]):
                temp_X = [np.concatenate((user_f[k][i], serv_f[k][j])) for k in range(len(gcn_feature))]
                if train_matrix[i][j] != 0:
                    X_train.append(temp_X)
                    Y_train.append(train_matrix[i][j])
                else:
                    X_pred.append(temp_X)

        return np.array(X_train), np.array(Y_train), np.array(X_pred)

def transformer_train(k, X_train, Y_train, custom_loss_fn):
    with tf.device('gpu'):
        model = build_model(
            input_shape=X_train.shape[1:],
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.4,
        )
        model.compile(
            loss=custom_loss_fn,
            metrics=['mae'],
            optimizer=keras.optimizers.Adam(learning_rate=1e-4)
        )

        callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

        start_time = time.time()
        history = model.fit(
            X_train, Y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )
        train_time = time.time() - start_time
        plot_loss(history)
        return model, train_time

def transformer_predict(model, k, X_pred, data_org, data_org_indicator, outlier_dict, data_train):
    with tf.device('cpu'):
        pred = model.predict(X_pred)
        pred_transformer = fill_matrix(k, pred)

        for pct, mask in outlier_dict.items():
            mae, rmse, _ = cal_metric(data_org[:,:,k], data_org_indicator[:,:,k], mask[:,:,k], pred_transformer)
            print(f"{pct}% Outliers Removed -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return pred_transformer


def run_pte(gcn_features, data_train, percent, k, outlier_dict, data_org, data_org_indicator):
    print("Preparing input...")
    start = time.time()
    X_train, Y_train, X_pred = transformer_inputs(gcn_features, data_train[:,:,k])
    print(f"Input preparation time: {time.time() - start:.2f}s")

    np.save(f'./data_files/X_train_{percent}.npy', X_train)
    np.save(f'./data_files/Y_train_{percent}.npy', Y_train)
    np.save(f'./data_files/X_pred_{percent}.npy', X_pred)

    X_train = np.load(f'./data_files/X_train_{percent}.npy')
    Y_train = np.load(f'./data_files/Y_train_{percent}.npy')
    X_pred = np.load(f'./data_files/X_pred_{percent}.npy')

    model, t_train_time = transformer_train(k, X_train[:, 56:, :], Y_train, cauchy_loss_fn)

    pred_matrix, t_pred_time = transformer_predict(
        model, k, X_pred[:, 56:, :],
        data_org, data_org_indicator,
        outlier_dict, data_train
    ), 0  # Replace 0 with actual timing if needed

    print(f"Total Training Time: {t_train_time:.2f}s, Prediction Time per QoS entry: {t_pred_time:.8f}s")
    return pred_matrix
