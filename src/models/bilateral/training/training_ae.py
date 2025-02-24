import numpy as np
import pandas as pd
from src.display import plot_history
from src.misc import set_seed
from src.models.bilateral.get_model import get_ae_model
from src.models.bilateral.training.training import get_best_val_loss
from src.preprocessing.tabular_preprocessing import preprocess_data_for_ae


import tensorflow as tf


def train_ae(
    bilateral_model, train_df, valid_df, test_df,
    batch_size, seed, max_epochs, patience=20, verbose=1
):
    # Set Seed
    set_seed(seed)

    # Get feature col information from model
    input_cols = list(bilateral_model.input_cols.numpy().astype('str'))
    le_label = bilateral_model.le_label.numpy().decode("utf-8")
    re_label = bilateral_model.re_label.numpy().decode("utf-8")

    # Preprocess data
    train_X = preprocess_data_for_ae(train_df, input_cols, le_label, re_label)
    valid_X = preprocess_data_for_ae(valid_df, input_cols, le_label, re_label)

    # Create ae model from bilateral model
    temp_ae_model = get_ae_model(bilateral_model)

    # Train ae model
    temp_ae_model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True,
    )
    history = temp_ae_model.fit(
        x=train_X, y=train_X, validation_data=(valid_X, valid_X),
        batch_size=batch_size, epochs=max_epochs, shuffle=True,
        callbacks=[es], verbose=verbose,
    )
    plot_history(history, metrics_list=["loss"], title="Autoencoder Training History")
    best_val_performance, best_val_performance_wo_sparsity, best_epoch = get_best_val_loss(history, metric="mse")

    # Freeze layers
    bilateral_model.encoder.trainable = False
    bilateral_model.decoder.trainable = False

    return best_val_performance, best_val_performance_wo_sparsity, best_epoch


def evaluate_ae(bilateral_model, train_df, valid_df, test_df):
    # Get feature col information from model
    input_cols = list(bilateral_model.input_cols.numpy().astype('str'))
    le_label = bilateral_model.le_label.numpy().decode("utf-8")
    re_label = bilateral_model.re_label.numpy().decode("utf-8")

    # Create ae model from bilateral model
    temp_ae_model = get_ae_model(bilateral_model)

    # Preprocess data
    train_X = preprocess_data_for_ae(train_df, input_cols, le_label, re_label)
    valid_X = preprocess_data_for_ae(valid_df, input_cols, le_label, re_label)
    test_X = preprocess_data_for_ae(test_df, input_cols, le_label, re_label)

    # Evaluate ae model
    train_X_pred = temp_ae_model.predict(train_X)
    valid_X_pred = temp_ae_model.predict(valid_X)
    test_X_pred = temp_ae_model.predict(test_X)
    def mse(actual, pred):
        return np.mean(np.square(actual-pred))
    train_mse = mse(train_X, train_X_pred)
    valid_mse = mse(valid_X, valid_X_pred)
    test_mse = mse(test_X, test_X_pred)
    perf_df = pd.DataFrame({"Train": [train_mse], "Valid": [valid_mse], "Test": [test_mse]})
    perf_df.index = ["Autoencoder MSE"]

    # Output predictions
    all_X = np.concatenate((train_X, valid_X, test_X), axis=0)
    all_X_pred = np.concatenate((train_X_pred, valid_X_pred, test_X_pred), axis=0)
    array = np.concatenate((all_X, all_X_pred), axis=1)
    colnames = input_cols + [feat_col+"_pred" for feat_col in input_cols]
    pred_df = pd.DataFrame(array, columns=colnames)
    split_labels = (
        ["train" for i in range(len(train_X))] + ["valid" for i in range(len(valid_X))] +
        ["test" for i in range(len(test_X))])
    pred_df["split"] = split_labels
    return perf_df, pred_df