from keras.models import Model, Sequential
from keras import layers
from keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.engine.functional import Functional # for correct type hints

from typing import List, Tuple, Union

from .keras_models import get_base_model

def build_model(
    model_name: str, 
    input_shape: Union[Tuple[int, int], Tuple[int, int, int]], 
    top_layers: List[layers.Layer], 
    lr: float
    ) -> Tuple[Sequential, Functional]:
    """
    Create a model for transfer learning using Keras pre-trained model.
    
    Since, the base model is inside a wrapper and its individual, internal 
    layers cannot be accessed through the main model, hence a reference to it 
    is also returned. 
    """ 
    # expand the input shape
    input_shape = (*input_shape, 3)
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
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model, base_model


def unfreeze_model(model: Model, base_model: Functional, n: int, lr: float) -> None:
    """ 
    Unfreeze the last `n` layers in the base model. 
    """ 
    # unfreeze the top `num_layers_unfreeze` layers while leaving BatchNorm layers frozen
    for layer in base_model.layers[-200:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    # re-compile
    model.compile(
        optimizer=Adam(learning_rate=lr), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )