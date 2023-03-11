import tensorflow as tf
from keras import layers
from keras.optimizer_v2.adam import Adam
from keras.callbacks import History

import argparse
from pathlib import Path
from typing import Tuple

import keras_transfer_learning as ktl

# create an ArgumentParser object
parser = argparse.ArgumentParser(description="""
    Fine-tune one of the Keras pre-trained CNN models.
""")

# add some arguments
parser.add_argument(
    '-d', '--dataset',
    type=str,
    required=True,
    help='Name of Keras dataset: mnist, fmnist, or cifar.'
)
parser.add_argument(
    '-m',
    '--model',
    type=str,
    required=True,
    help='Name of Keras dataset: mnist, fmnist, or cifar.'
)
parser.add_argument(
    '-ie',
    '--initial-epochs',
    type=int,
    default=20,
    help='Number of epochs to train the new layers of the model (default: 20).'
)
parser.add_argument(
    '-fte',
    '--fine-tune-epochs',
    type=int,
    default=10,
    help='Number of epochs to train the unfozen layers of the base model (default: 10).'
)
parser.add_argument(
    '-n',
    '--n-layers-unfreeze',
    type=int,
    default=20,
    help='Number of layers of the base model to unfreeze for the fine-tuning step (default: 20).'
)
parser.add_argument(
    '-bs',
    '--batch-size',
    type=int,
    default=128,
    help='Batch size used in both the initial and fine-tuning steps (default: 128).'
)
parser.add_argument(
    '-sb',
    '--shuffle-buffer',
    type=int,
    default=256,
    help='Size of the shuffle buffer used in the tf.data pipeline (default: 256).'
)
parser.add_argument(
    '-ilr',
    '--initial-lr',
    type=float,
    default=1e-3,
    help='Learning rate used to train the new layers of the models (default: 1e-3).'
)
parser.add_argument(
    '-ftlr',
    '--fine-tune-lr',
    type=float,
    default=1e-5,
    help='Learning rate used to fine-tune the model (default: 1e-5).'
)


def main(
    dataset: str,
    model_name: str,
    initial_epochs: int = 20,
    fine_tune_epochs: int = 10,
    n: int = 20,
    batch_size: int = 128,
    shuffle_buffer: int = 256,
    ilr: float = 1e-3,
    ftlr: float = 1e-5,
) -> Tuple[History, History]:
    (x_train, y_train), (x_test, y_test), key = ktl.load_dataset(dataset)
    top_layers = [
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ]
    model, base_model = ktl.build_model(
        model_name=model_name,
        input_shape=(32, 32),
        top_layers=top_layers
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    def augmentation_layer(x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, dtype=tf.float32)
        x = tf.image.random_flip_left_right(x)

        return x

    train_dataset, test_dataset = ktl.get_data_generators(
        x_train, y_train, x_test, y_test, augmentation_layer, batch_size, shuffle_buffer)
    initial_history = model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=test_dataset,
    )
    ktl.unfreeze_model(model, base_model, n)
    model.compile(
        optimizer=Adam(learning_rate=ftlr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    total_epochs = initial_epochs + fine_tune_epochs
    ft_history = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=initial_history.epoch[-1]+1,
        validation_data=test_dataset
    )

    dir = Path('./models/').resolve()
    if not dir.exists():
        dir.mkdir()
    model.save(dir / f'{model_name}_{dataset}_{total_epochs}epochs')

    return initial_history, ft_history


if __name__ == '__main__':
    args = parser.parse_args()
    initial_history, ft_history = main(
        dataset=args.dataset,
        model_name=args.model,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        n=args.n_layers_unfreeze,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        ilr=args.initial_lr,
        ftlr=args.fine_tune_lr
    )
    ktl.plot_history(initial_history)
    ktl.plot_history(ft_history)
