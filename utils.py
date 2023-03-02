import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras
from keras.models import Model
from keras.callbacks import History

from typing import Callable, Dict, List, Literal, Tuple, Union

def plot_history(history: History) -> None:
    epochs = len(history.history['loss'])
    x = range(1, epochs+1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(x, history.history['loss'], label='Loss')
    axs[0].plot(x, history.history['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')

    axs[1].plot(x, history.history['accuracy'], label='Accuracy')
    axs[1].plot(x, history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training and Validation Accuracy')
    
    for i in (0, 1):
        axs[i].set_xticks(np.arange(1, epochs+1, 1))
        axs[i].set_xlim(1, epochs)
        axs[i].legend()

    plt.show()
    
def plot_confusion_matrix(model: Model, x_train: np.ndarray, y_train: np.ndarray, labels: List[str]) -> None:
    y_pred = np.argmax(model.predict(x_train), axis=1) 
    y_true = np.argmax(y_train, axis=1) 
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    

def load_dataset(name: Literal['mnist', 'fmnist', 'cifar']) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[int, str]]:
    (x_train, y_train), (x_test, y_test) = {
        'mnist': tf.keras.datasets.mnist.load_data,
        'fmnist': tf.keras.datasets.fashion_mnist.load_data,
        'cifar': tf.keras.datasets.cifar10.load_data,
    }[name]()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    if name in ('mnist', 'fmnist'):
        x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
        x_test = np.repeat(x_test[..., np.newaxis], 3, -1)
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))
    
    key = {
        'mnist': {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
        },
        'fmnist': {
            0 : "T-shirt/top",
            1 : "Trouser",
            2 : "Pullover",
            3 : "Dress",
            4 : "Coat",
            5 : "Sandal",
            6 : "Shirt",
            7 : "Sneaker",
            8 : "Bag",
            9 : "Ankle boot"
        },
        'cifar': {
            0: 'airplane', 
            1: 'automobile', 
            2: 'bird', 
            3: 'cat', 
            4: 'deer', 
            5: 'dog', 
            6: 'frog', 
            7: 'horse', 
            8: 'ship', 
            9: 'truck'
        },
    }[name]
    
    return (x_train, y_train), (x_test, y_test), key

def get_data_generators(x_train: np.ndarray, y_train: np.ndarray, 
                        x_test: np.ndarray, y_test: np.ndarray,
                        augmentation_layer: Callable[[tf.Tensor], tf.Tensor],
                        batch_size: int = 128, shuffle_buffer_size: int = 256
                        ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.map(lambda x, y: (augmentation_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset