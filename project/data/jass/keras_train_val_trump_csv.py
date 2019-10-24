# HSLU
#
# Created by Thomas Koller on 01.10.19
#
"""
Example for learning trump selection in Keras.
"""

import argparse
import numpy as np
import pandas as pd
from tensorflow import keras
import logging


def train(files_train: [str], files_val: [str], nr_epochs: int, batch_size: int, log_dir: str):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # read the data directly into panda
    logger.info('Reading data')

    data_pd = pd.concat([pd.read_csv(f, header=None) for f in files_train], sort=False)
    data_train = data_pd.values

    logger.info('training data: read {} data rows'.format(data_train.shape[0]))

    data_pd = pd.concat([pd.read_csv(f, header=None) for f in files_val], sort=False)
    data_val = data_pd.values
    logger.info('validation data: read {} data rows'.format(data_val.shape[0]))

    # x data is the set of cards in hand (36) and indication if the partner player has declined to select
    # trump already (geschoben)
    x = data_train[:,0:37].astype(np.float32)

    # y data is the action 0..5 for trump and 6 for "schieben"
    y = keras.utils.to_categorical(data_train[:,-1], num_classes=7)

    x_val = data_val[:,0:37].astype(np.float32)
    y_val = keras.utils.to_categorical(data_val[:,-1], num_classes=7)

    model = keras.Sequential()
    model.add(keras.layers.Dense(37 * 16, activation='relu', input_shape=[37]))
    model.add(keras.layers.Dense(37 * 16, activation='relu'))
    model.add(keras.layers.Dense(37 * 16, activation='relu'))
    model.add(keras.layers.Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    tensorboard = \
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            batch_size=batch_size,
            write_images=False,
            write_graph=True,
            write_grads=False)

    model.fit(x, y, validation_data=(x_val, y_val),
              epochs=nr_epochs, batch_size=batch_size,
              shuffle=True,
              callbacks=[tensorboard])


def main():
    parser = argparse.ArgumentParser(description='Train DNN for trump')
    parser.add_argument('--train_files', type=str, nargs='+', required=True, help='Files for training')
    parser.add_argument('--val_files', type=str, nargs='+', required=True, help='Files for validation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory for tensorboard')

    arg = parser.parse_args()

    train(arg.train_files, arg.val_files, arg.epochs, arg.batch_size, arg.log_dir)


if __name__ == "__main__":
    main()
