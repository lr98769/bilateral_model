import tensorflow as tf

import numpy as np
import pandas as pd
from src.display import plot_history
from src.misc import set_seed
from src.models.bilateral.get_model import get_final_classifier_model
from src.models.bilateral.training.training import get_best_val_loss
from src.preprocessing.tabular_preprocessing import preprocess_data_for_fc

def train_fc(
    bilateral_model, train_df, valid_df, test_df, 
    batch_size, max_epochs, seed, patience=20, verbose=1, 
):
    # Set Seed
    set_seed(seed)

    # Get feature col information from model
    input_cols = list(bilateral_model.input_cols.numpy().astype('str'))
    output_cols = list(bilateral_model.output_cols.numpy().astype('str'))
    le_label = bilateral_model.le_label.numpy().decode("utf-8")
    re_label = bilateral_model.re_label.numpy().decode("utf-8")
    
    # Preprocess data
    train_X, train_Y = preprocess_data_for_fc(train_df, input_cols, output_cols, le_label, re_label)
    valid_X, valid_Y = preprocess_data_for_fc(valid_df, input_cols, output_cols, le_label, re_label)

    # Create ic model from bilateral model
    temp_fc_model = get_final_classifier_model(bilateral_model)

    # Train ic model
    temp_fc_model.compile(
        optimizer="adam", loss="binary_crossentropy",  metrics=["binary_crossentropy", "binary_accuracy"])
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True,
    )
    history = temp_fc_model.fit(
        x=train_X, y=train_Y, validation_data=(valid_X, valid_Y),
        batch_size=batch_size, epochs=max_epochs, shuffle=True,
        callbacks=[es], verbose=verbose, 
    )
    plot_history(history, metrics_list=["loss", "binary_accuracy"], title="Final Classifier Training History")
    best_val_performance, best_val_performance_wo_sparsity, best_epoch = get_best_val_loss(history, metric="binary_crossentropy")

    # Freeze layers
    bilateral_model.final_classifier.trainable = False
    
    return best_val_performance, best_val_performance_wo_sparsity, best_epoch

def evaluate_fc(bilateral_model, train_df, valid_df, test_df):
    # Create ic model from bilateral model
    temp_fc_model = get_final_classifier_model(bilateral_model)

    # Get feature col information from model
    input_cols = list(bilateral_model.input_cols.numpy().astype('str'))
    output_cols = list(bilateral_model.output_cols.numpy().astype('str'))
    le_label = bilateral_model.le_label.numpy().decode("utf-8")
    re_label = bilateral_model.re_label.numpy().decode("utf-8")
    
    # Preprocess data
    train_X, train_Y = preprocess_data_for_fc(train_df, input_cols, output_cols, le_label, re_label)
    valid_X, valid_Y = preprocess_data_for_fc(valid_df, input_cols, output_cols, le_label, re_label)
    test_X, test_Y = preprocess_data_for_fc(test_df, input_cols, output_cols, le_label, re_label)

    # Evaluate ae model
    train_Y_pred = temp_fc_model.predict(train_X)
    valid_Y_pred = temp_fc_model.predict(valid_X)
    test_Y_pred = temp_fc_model.predict(test_X)
    def binary_accuracy(actual, pred):
        return np.mean(
            tf.keras.metrics.binary_accuracy(actual, pred))
    train_bin_acc = binary_accuracy(train_Y, train_Y_pred)
    valid_bin_acc = binary_accuracy(valid_Y, valid_Y_pred)
    test_bin_acc = binary_accuracy(test_Y, test_Y_pred)
    def accuracy(actual, pred):
        pred = pred > 0.5
        return np.mean(np.all(actual == pred, axis=-1))
    train_acc = accuracy(train_Y, train_Y_pred)
    valid_acc = accuracy(valid_Y, valid_Y_pred)
    test_acc = accuracy(test_Y, test_Y_pred)
    def binary_crossentropy(actual, pred):
        return np.mean(
            tf.keras.metrics.binary_crossentropy(actual, pred))
    train_crossentropy = binary_crossentropy(train_Y, train_Y_pred)
    valid_crossentropy = binary_crossentropy(valid_Y, valid_Y_pred)
    test_crossentropy = binary_crossentropy(test_Y, test_Y_pred)

    def convert_label_to_index(labels):
        return [np.sum(cur_label, axis=-1)-1 for cur_label in labels]
            
    def get_class_accuracies(actual, pred):
        value_list, label_list = [], []
        pred = pred > 0.5
        correctness = np.all(actual == pred, axis=-1)
        class_label = convert_label_to_index(actual)
        split_df = pd.DataFrame({"actual":class_label, "correct?":correctness})
        for i, class_label in enumerate(output_cols):
            class_df = split_df[split_df["actual"]==i]
            class_acc = class_df["correct?"].mean()
            value_list.append(class_acc)
            label_list.append(f"Accuracy {class_label}")
            output_dict = class_acc
            class_size = len(class_df)
            value_list.append(class_size/len(pred))
            label_list.append(f"{class_label} Proportion")
        return value_list, label_list

    train_class_acc = get_class_accuracies(train_Y, train_Y_pred)
    valid_class_acc = get_class_accuracies(valid_Y, valid_Y_pred)
    test_class_acc = get_class_accuracies(test_Y, test_Y_pred)
    
    perf_df = pd.DataFrame({
        "Train": [train_acc, train_crossentropy, train_bin_acc]+train_class_acc[0], 
        "Valid": [valid_acc, valid_crossentropy, valid_bin_acc]+valid_class_acc[0], 
        "Test": [test_acc, test_crossentropy, test_bin_acc]+test_class_acc[0]})
    perf_df.index = ["Final Classifier Accuracy", "Final Classifier Crossentropy", "Final Classifier Bin Accuracy"] + test_class_acc[1]
    perf_df = perf_df.sort_index().round(3)
    
    # Output predictions
    all_Y = np.concatenate((train_Y, valid_Y, test_Y), axis=0)
    all_Y_pred = np.concatenate((train_Y_pred, valid_Y_pred, test_Y_pred), axis=0)
    array = np.concatenate((all_Y, all_Y_pred), axis=1)
    colnames = output_cols + [final_col+"_pred" for final_col in output_cols]
    pred_df = pd.DataFrame(array, columns=colnames)
    split_labels = (
        ["train" for i in range(len(train_Y))] + ["valid" for i in range(len(valid_Y))] + 
        ["test" for i in range(len(test_Y))])
    pred_df["split"] = split_labels
    return perf_df, pred_df
