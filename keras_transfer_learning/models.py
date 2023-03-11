from keras.models import Model, Sequential
from keras import layers
from keras.optimizer_v2.adam import Adam
# for correct type hints
from tensorflow.python.keras.engine.functional import Functional

from typing import Any, List, Optional, Tuple, Union

from .keras_models import get_base_model


def build_model(
    model_name: str,
    input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    top_layers: List[layers.Layer],
    optimizer: Any,
    learning_rate: Optional[float] = None
) -> Tuple[Sequential, Functional]:
    """
    Create a model for transfer learning using Keras pre-trained model.

    Since, the base model is inside a wrapper within the main model, internal 
    layers cannot be accessed through the main model API. Reference to the 
    base model is also returned to allow its layers to be unforzen. 
    """
    input_shape = input_shape if len(input_shape) == 3 else (*input_shape, 3)
    # get the base, pre-trained model
    base_model, preprocess_input = get_base_model(model_name, input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    # build the main model
    model = Sequential(name=model_name+'_transfer')
    model.add(layers.Lambda(preprocess_input, input_shape=input_shape))
    model.add(base_model)
    for layer in top_layers:
        model.add(layer)
    # compile
    opt = (
        optimizer if learning_rate is None
        else optimizer(learning_rate=learning_rate)
    )
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def unfreeze_model(model: Model, base_model: Functional, n: int, learning_rate: float) -> None:
    """ 
    Unfreeze the last `n` layers in the base model. 
    """
    # unfreeze the top `n` layers while leaving BatchNormalization layers frozen
    for layer in base_model.layers[-200:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    # re-compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
