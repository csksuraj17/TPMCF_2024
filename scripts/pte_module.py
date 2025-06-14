
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.notebook import tqdm as tqdm_notebook
import time
import copy
import os
from csfe_module import fill_matrix, cal_metric, plot_loss


def custom_cauchy(y_true, y_pred):
    loss = tf.math.log(1+tf.square(y_pred - y_true))
    return loss

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = embed_dim // num_heads
        self.dropout = dropout

        # Layers to project inputs to Q, K, V
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)

        # Output projection
        self.out_dense = layers.Dense(embed_dim)

        # Dropout
        self.attn_dropout = layers.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, proj_dim).
        Transpose the result to shape (batch_size, num_heads, seq_len, proj_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        # inputs: (batch_size, seq_len, embed_dim)
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Split into heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        score = tf.matmul(query, key, transpose_b=True)  # (B, H, L, L)
        dk = tf.cast(self.proj_dim, tf.float32)
        scaled_score = score / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        attention = tf.matmul(weights, value)  # (B, H, L, D/H)

        # Concatenate heads
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (B, L, H, D/H)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        # Final linear layer
        output = self.out_dense(concat_attention)  # (B, L, D)

        return output

def transformer_inputs(gcn_feature, train_matrix):
    with tf.device('gpu'):
        user_f, serv_f = tf.split(gcn_feature, [142, 4500], 1)

        X_train = []
        Y_train = []
        X_pred = []

        for i in tqdm_notebook(range(train_matrix.shape[0])):
            for j in range(train_matrix.shape[1]):
                temp_X = []
                for k in range(len(gcn_feature)):
                    temp_X.append(np.concatenate((user_f[k][i], serv_f[k][j])))
                if train_matrix[i][j] != 0:
                    X_train.append(temp_X)
                    Y_train.append(train_matrix[i][j])
                else:
                    X_pred.append(temp_X)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_pred = np.array(X_pred)

    return X_train, Y_train, X_pred

def transformer_encoder(inputs, mha_layer, head_size, num_heads, ff_dim, dropout=0):
    attn_output = mha_layer(inputs)
    out1 = layers.LayerNormalization(epsilon=1e-6)(attn_output + inputs)

    x = tf.transpose(out1, perm=[0, 2, 1])
    ffn_output = layers.Conv1D(filters=ff_dim, kernel_size=3, activation="relu", padding='same')(x)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    ffn_output = layers.Conv1D(filters=inputs.shape[-2], kernel_size=1)(ffn_output)
    ffn_output = tf.transpose(ffn_output, perm=[0, 2, 1])
    out2 = layers.LayerNormalization(epsilon=1e-6)(ffn_output + out1)
    return out2

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    x = inputs
    inputs = keras.Input(shape=input_shape)
    mha_layer = MultiHeadSelfAttention(embed_dim=head_size, num_heads=num_heads, dropout=dropout)
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, mha_layer, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

def transformer_train(k, X_train, Y_train):
    with tf.device('gpu'):
        input_shape = X_train.shape[1:]
        model = build_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.4,
        )

        model.compile(
            loss=custom_cauchy,
            metrics=['mae'],
            optimizer=keras.optimizers.Adam(learning_rate=0.0001)
        )
        model.summary()

        callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

        start_time = time.time()
        history = model.fit(
            X_train,
            Y_train,
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

def train_transformer(gcn_features, data_train, percent, k=63):
    print("Preparing input...")
    start = time.time()
    X_train, Y_train, X_pred = transformer_inputs(gcn_features, data_train[:,:,k])
    print("Input preparation time: {:.2f}s".format(time.time() - start))

    base_path = './data_files/{}_8time_raw_rt_{}.npy'
    np.save(base_path.format('X_train', percent), X_train)
    np.save(base_path.format('Y_train', percent), Y_train)
    np.save(base_path.format('X_pred', percent), X_pred)

    X_train = np.load(base_path.format('X_train', percent))
    Y_train = np.load(base_path.format('Y_train', percent))
    X_pred = np.load(base_path.format('X_pred', percent))

    xt = copy.deepcopy(X_train[:, 56:, :])
    xp = copy.deepcopy(X_pred[:, 56:, :])

    model, train_time = transformer_train(k, xt, Y_train)
  
    pred_matrix = transformer_predict(
        model, k, xp, data_org, data_org_indicator,
        {
            0: no_outlier,
            2: outlier2,
            4: outlier4,
            6: outlier6,
            8: outlier8,
            10: outlier10
        },
        data_train
    )

    print("Total Training Time: {:.2f}s, Prediction Time per QoS entry: {:.8f}s".format(t_train_time, t_pred_time))
    return pred_matrix
