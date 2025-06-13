
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

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    out1 = layers.LayerNormalization(epsilon=1e-6)(attn_output + inputs)

    x = tf.transpose(out1, perm=[0, 2, 1])
    ffn_output = layers.Conv1D(filters=ff_dim, kernel_size=3, activation="relu", padding='same')(x)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    ffn_output = layers.Conv1D(filters=inputs.shape[-2], kernel_size=1)(ffn_output)
    ffn_output = tf.transpose(ffn_output, perm=[0, 2, 1])
    out2 = layers.LayerNormalization(epsilon=1e-6)(ffn_output + out1)
    return out2

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
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
