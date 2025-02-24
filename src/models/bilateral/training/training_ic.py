import numpy as np
import tensorflow as tf
import pandas as pd

from src.misc import set_seed
from src.preprocessing.tabular_preprocessing import preprocess_data_for_ic
from src.models.bilateral.get_model import get_intermediate_classifier_model
from src.display import plot_history
from src.models.bilateral.training.training import get_best_val_loss

def train_ic(
    bilateral_model, train_df, valid_df, test_df, 
    batch_size, max_epochs, seed, patience=20, verbose=1
):
    # Set Seed
    set_seed(seed)

    # Get feature col information from model
    input_cols = list(bilateral_model.input_cols.numpy().astype('str'))
    intermediate_cols = list(bilateral_model.intermediate_cols.numpy().astype('str'))
    intermediate_col_groups = list(bilateral_model.intermediate_col_groups.numpy().astype('str'))
    intermediate_col_sizes = list(bilateral_model.intermediate_col_sizes.numpy())
    num_intermediate_col_groups = len(intermediate_col_groups)
    le_label = bilateral_model.le_label.numpy().decode("utf-8")
    re_label = bilateral_model.re_label.numpy().decode("utf-8")

    # Preprocess data
    train_X, train_Y = preprocess_data_for_ic(train_df, input_cols, intermediate_cols, intermediate_col_sizes, le_label, re_label)
    valid_X, valid_Y = preprocess_data_for_ic(valid_df, input_cols, intermediate_cols, intermediate_col_sizes, le_label, re_label)

    # Create ic model from bilateral model
    temp_ic_model = get_intermediate_classifier_model(bilateral_model)

    # Train ic model
    temp_ic_model.compile(
        optimizer="adam", 
        loss=["categorical_crossentropy" for i in range(num_intermediate_col_groups)],
        metrics=[["categorical_crossentropy", "accuracy"] for i in range(num_intermediate_col_groups)]
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True,
    )
    history = temp_ic_model.fit(
        x=train_X, y=train_Y, validation_data=(valid_X, valid_Y),
        batch_size=batch_size, epochs=max_epochs, shuffle=True,
        callbacks=[es], verbose=verbose, 
    )
    plot_history(
        history, metrics_list=["loss"]+[
            metric for metric in history.history.keys() if (("val" not in metric) and ("accuracy" in metric))], 
        title="Intermediate Classifier Training History"
    )
    new_loss = None
    for metric, metric_values in history.history.items():
        if (("val" in metric) and ("categorical_crossentropy" in metric)):
            if new_loss is None:
                new_loss = np.array(metric_values)
            else:
                new_loss = new_loss+np.array(metric_values)
    history.history["val_categorical_crossentropy"] = new_loss
    best_val_performance, best_val_performance_wo_sparsity, best_epoch = get_best_val_loss(
        history, metric="categorical_crossentropy")

    # Freeze layers
    bilateral_model.intermediate_classifier.trainable = False
    
    return best_val_performance, best_val_performance_wo_sparsity, best_epoch

def evaluate_ic(bilateral_model, train_df, valid_df, test_df):
    # Get feature col information from model
    input_cols = list(bilateral_model.input_cols.numpy().astype('str'))
    intermediate_cols = list(bilateral_model.intermediate_cols.numpy().astype('str'))
    intermediate_col_sizes = list(bilateral_model.intermediate_col_sizes.numpy())
    intermediate_col_groups = list(bilateral_model.intermediate_col_groups.numpy().astype('str'))
    le_label = bilateral_model.le_label.numpy().decode("utf-8")
    re_label = bilateral_model.re_label.numpy().decode("utf-8")
    
    # Create ic model from bilateral model
    temp_ic_model = get_intermediate_classifier_model(bilateral_model)
    
    # Preprocess data
    train_X, train_Y = preprocess_data_for_ic(train_df, input_cols, intermediate_cols, intermediate_col_sizes, le_label, re_label)
    valid_X, valid_Y = preprocess_data_for_ic(valid_df, input_cols, intermediate_cols, intermediate_col_sizes, le_label, re_label)
    test_X, test_Y = preprocess_data_for_ic(test_df, input_cols, intermediate_cols, intermediate_col_sizes, le_label, re_label)
    
    # Evaluate ae model
    train_Y_pred = temp_ic_model.predict(train_X)
    valid_Y_pred = temp_ic_model.predict(valid_X)
    test_Y_pred = temp_ic_model.predict(test_X)
    def per_intermediate_class_score(actual, pred, score_func):
        all_score = []
        all_score_cols = []
        index = 0
        for col_group, col_size in zip(intermediate_col_groups, intermediate_col_sizes):
            col_actual = actual[index]
            col_pred = pred[index]
            # print(col_actual, col_pred)
            col_score = score_func(col_actual, col_pred)
            index += 1
            all_score.append(col_score)
            all_score_cols.append(f"{col_group} {score_func.__name__.capitalize()}")
        return all_score, all_score_cols
    def accuracy(actual, pred):
        actual_index = np.argmax(actual, axis=-1)
        pred_index = np.argmax(pred, axis=-1)
        acc = np.mean(actual_index==pred_index)
        return acc
    train_ic_accuracy, _ = per_intermediate_class_score(train_Y, train_Y_pred, accuracy)
    valid_ic_accuracy, _ = per_intermediate_class_score(valid_Y, valid_Y_pred, accuracy)
    test_ic_accuracy, acc_cols = per_intermediate_class_score(test_Y, test_Y_pred, accuracy)
    def f1_score(actual, pred):
        actual_index = np.argmax(actual, axis=-1)
        pred_index = np.argmax(pred, axis=-1)
        from sklearn.metrics import f1_score
        f1 = f1_score(actual_index, pred_index, average='macro')
        return f1
    train_ic_f1, _ = per_intermediate_class_score(train_Y, train_Y_pred, f1_score)
    valid_ic_f1, _ = per_intermediate_class_score(valid_Y, valid_Y_pred, f1_score)
    test_ic_f1, f1_cols = per_intermediate_class_score(test_Y, test_Y_pred, f1_score)
    def categorical_crossentropy(actual, pred):
        return np.mean(
            tf.keras.metrics.categorical_crossentropy(actual, pred))
    train_crossentropy, _ = per_intermediate_class_score(train_Y, train_Y_pred, categorical_crossentropy)
    valid_crossentropy, _ = per_intermediate_class_score(valid_Y, valid_Y_pred, categorical_crossentropy)
    test_crossentropy, crossentropy_cols = per_intermediate_class_score(test_Y, test_Y_pred, categorical_crossentropy)
    perf_df = pd.DataFrame({
        "Train": train_crossentropy+train_ic_accuracy+train_ic_f1, 
        "Valid": valid_crossentropy+valid_ic_accuracy+valid_ic_f1, 
        "Test": test_crossentropy+test_ic_accuracy+test_ic_f1})
    perf_df.index = crossentropy_cols+acc_cols+f1_cols

    # Output predictions
    # test_Y from [num_groups, num_samples, num_cols]
    train_Y = np.concatenate(train_Y, axis=1)
    train_Y_pred = np.concatenate(train_Y_pred, axis=1)
    valid_Y = np.concatenate(valid_Y, axis=1)
    valid_Y_pred = np.concatenate(valid_Y_pred, axis=1)
    test_Y = np.concatenate(test_Y, axis=1)
    test_Y_pred = np.concatenate(test_Y_pred, axis=1)
    # print(test_Y.shape)
    all_Y = np.concatenate((train_Y, valid_Y, test_Y), axis=0)
    all_Y_pred = np.concatenate((train_Y_pred, valid_Y_pred, test_Y_pred), axis=0)
    array = np.concatenate((all_Y, all_Y_pred), axis=1)
    colnames = intermediate_cols + [inter_col+"_pred" for inter_col in intermediate_cols]
    pred_df = pd.DataFrame(array, columns=colnames)
    split_labels = (
        ["train" for i in range(len(train_Y))] + ["valid" for i in range(len(valid_Y))] + 
        ["test" for i in range(len(test_Y))])
    pred_df["split"] = split_labels
    return perf_df, pred_df
