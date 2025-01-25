import numpy as np
import os
import tensorflow as tf




def load_dataset(train_data_file,train_labels_file,test_data_file,test_labels_file):
    if all(os.path.exists(file) for file in [train_data_file, train_labels_file, test_data_file, test_labels_file]):
        X_train = np.load(train_data_file)
        y_train = np.load(train_labels_file)
        X_test = np.load(test_data_file)
        y_test = np.load(test_labels_file)
        print("Loaded preprocessed train and test data from disk.")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1000).batch(32)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
    return train_dataset, test_dataset