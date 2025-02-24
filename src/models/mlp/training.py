import numpy as np

from keras import Sequential, Input
from keras.layers import Dense
import tensorflow as tf

from src.misc import set_seed
from src.display import plot_history
from src.preprocessing.tabular_preprocessing import preprocess_data_for_baseline

def get_baseline_model(width, num_layers, num_input_cols, num_output_cols):
    model = Sequential()
    model.add(Input(shape=(num_input_cols,)))
    for i in range(num_layers-1):
        model.add(Dense(width, activation="relu"))
    model.add(Dense(num_output_cols,  activation="softmax"))
    return model

def get_best_val_loss(history):
    best_epoch = np.argmin(history.history["val_loss"])
    best_val_loss = min(history.history["val_loss"])
    return best_val_loss, best_epoch

def train_baseline(
    baseline_model, train_df, valid_df, test_df, col_info, seed, 
    batch_size, max_epochs, patience=20, verbose=1, 
):
    # Set Seed
    set_seed(seed)

    # Get feature col information 
    input_cols = col_info["input_cols"]
    output_cols = col_info["output_cols"]
    le_label = col_info["le_label"]
    re_label = col_info["re_label"]
    
    # Preprocess data
    train_X, train_Y = preprocess_data_for_baseline(train_df, input_cols, output_cols, le_label, re_label)
    valid_X, valid_Y = preprocess_data_for_baseline(valid_df, input_cols, output_cols, le_label, re_label)

    # Train baseline model
    baseline_model.compile(
        optimizer="adam", loss="categorical_crossentropy",  metrics=["categorical_crossentropy", "accuracy"])
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True,
    )
    history = baseline_model.fit(
        x=train_X, y=train_Y, validation_data=(valid_X, valid_Y),
        batch_size=batch_size, epochs=max_epochs, shuffle=True,
        callbacks=[es], verbose=verbose, 
    )
    plot_history(history, metrics_list=["loss", "accuracy"], title="Baseline Training History")
    best_val_performance, best_epoch = get_best_val_loss(history)
    
    return best_val_performance, best_epoch