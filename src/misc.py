import pandas as pd
import tensorflow as tf
import numpy as np
from os.path import exists
from os import makedirs


def set_seed(seed):
    import os
    import random
    tf.config.experimental.enable_op_determinism()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def create_folder(fp):
    if not exists(fp):
        makedirs(fp)
        return True
    else:
        return False


def format_number(number, dp):
    return f"{number:.{dp}f}"


def get_mean_std_df(list_of_dfs, dp):
    matrix = np.array([df.values for df in list_of_dfs])
    mean_matrix = matrix.mean(axis=0)
    std_matrix = matrix.std(axis=0)
    combined_matrix = np.full(mean_matrix.shape, " ").astype('object')
    for row in range(combined_matrix.shape[0]):
        for col in range(combined_matrix.shape[1]):
            combined_matrix[row, col] = \
                format_number(mean_matrix[row, col], dp) + " ± " + format_number(std_matrix[row, col], dp)
    return pd.DataFrame(
        combined_matrix, columns=list_of_dfs[0].columns, index=list_of_dfs[0].index)