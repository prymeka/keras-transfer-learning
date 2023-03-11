import tensorflow as tf
from keras.models import Model
import tensorflow.keras.applications as m

from typing import Callable, Tuple, Union


def get_base_model(
    model_name: str,
    input_shape: Union[Tuple[float, float], Tuple[float, float, float]]
) -> Tuple[Model, Callable[[tf.Tensor], tf.Tensor]]:
    """
    Load one of the pre-trained keras models and corresponding function to 
    preprocess input data (note: not all models available in keras are 
    implemented).
    """
    kwargs = {
        include_top: False,
        weights: 'imagenet',
        input_shape:  input_shape if len(input_shape) == 3 else (*input_shape, 3),
    }
    model_name = model_name.lower()

    if model_name == 'xception':
        model = m.Xception(**kwargs)
        fn = m.xception.preprocess_input

    elif model_name == 'vgg16':
        model = m.VGG16(**kwargs)
        fn = m.vgg16.preprocess_input

    elif model_name == 'vgg19':
        model = m.VGG19(**kwargs)
        fn = m.vgg19.preprocess_input

    elif model_name == 'resnet50':
        model = m.ResNet50(**kwargs)
        fn = m.resnet.preprocess_input

    elif model_name == 'resnet101':
        model = m.ResNet101(**kwargs)
        fn = m.resnet.preprocess_input

    elif model_name == 'resnet152':
        model = m.ResNet152(**kwargs)
        fn = m.resnet.preprocess_input

    elif model_name == 'resnet50v2':
        model = m.ResNet50V2(**kwargs)
        fn = m.resnet_v2.preprocess_input

    elif model_name == 'resret101v2':
        model = m.ResNet101V2(**kwargs)
        fn = m.resnet_v2.preprocess_input

    elif model_name == 'resnet152v2':
        model = m.ResNet152V2(**kwargs)
        fn = m.resnet_v2.preprocess_input

    elif model_name == 'inceptionv3':
        model = m.InceptionV3(**kwargs)
        fn = m.inception_v3.preprocess_input

    elif model_name == 'inceptionresnetv2':
        model = m.InceptionResNetV2(**kwargs)
        fn = m.inception_resnet_v2.preprocess_input

    elif model_name == 'densenet121':
        model = m.DenseNet121(**kwargs)
        fn = m.densenet.preprocess_input

    elif model_name == 'densenet169':
        model = m.DenseNet169(**kwargs)
        fn = m.densenet.preprocess_input

    elif model_name == 'densenet201':
        model = m.DenseNet201(**kwargs)
        fn = m.densenet.preprocess_input

    elif model_name == 'efficientnetb0':
        model = m.EfficientNetB0(**kwargs)
        fn = m.efficientnet.preprocess_input

    elif model_name == 'efficientnetb1':
        model = m.EfficientNetB1(**kwargs)
        fn = m.efficientnet.preprocess_input

    elif model_name == 'efficientnetb2':
        model = m.EfficientNetB2(**kwargs)
        fn = m.efficientnet.preprocess_input

    elif model_name == 'efficientnetb3':
        model = m.EfficientNetB3(**kwargs)
        fn = m.efficientnet.preprocess_input

    elif model_name == 'efficientnetb4':
        model = m.EfficientNetB4(**kwargs)
        fn = m.efficientnet.preprocess_input

    elif model_name == 'efficientnetb5':
        model = m.EfficientNetB5(**kwargs)
        fn = m.efficientnet.preprocess_input

    elif model_name == 'efficientnetb6':
        model = m.EfficientNetB6(**kwargs)
        fn = m.efficientnet.preprocess_input

    elif model_name == 'efficientnetb7':
        model = m.EfficientNetB7(**kwargs)
        fn = m.efficientnet.preprocess_input

    elif model_name == 'efficientnetv2b0':
        model = m.EfficientNetV2B0(**kwargs)
        fn = m.efficientnet_v2.preprocess_input

    elif model_name == 'efficientnetv2b1':
        model = m.EfficientNetV2B1(**kwargs)
        fn = m.efficientnet_v2.preprocess_input

    elif model_name == 'efficientnetv2b2':
        model = m.EfficientNetV2B2(**kwargs)
        fn = m.efficientnet_v2.preprocess_input

    elif model_name == 'efficientnetv2b3':
        model = m.EfficientNetV2B3(**kwargs)
        fn = m.efficientnet_v2.preprocess_input

    elif model_name == 'efficientnetv2s':
        model = m.EfficientNetV2S(**kwargs)
        fn = m.efficientnet_v2.preprocess_input

    elif model_name == 'efficientnetv2m':
        model = m.EfficientNetV2M(**kwargs)
        fn = m.efficientnet_v2.preprocess_input

    elif model_name == 'efficientnetv2l':
        model = m.EfficientNetV2L(**kwargs)
        fn = m.efficientnet_v2.preprocess_input

    else:
        raise ValueError(f'Model name, {model_name}, not recognized.')

    return model, fn
