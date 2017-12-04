"""
SqueezeNet_v1.1 implementation
https://github.com/DeepScale/SqueezeNet
"""

from keras import Model, Input
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Concatenate, GlobalAveragePooling2D
from keras.regularizers import l2


def _build_fire_module(fire_id: int, squeeze_filters: int, expand_filtters: int, **conv_params):
    """
    Build a fire module
    :param fire_id: fire id, for layer names
    :param squeeze_filters: number of filters in squeeze layer
    :param expand_filtters: number of filters in expand layer
    :return: fire_module
    """

    conv_params.setdefault('kernel_initializer', 'he_normal')
    conv_params.setdefault('kernel_regularizer', l2(1e-4))

    def _module(x):
        x = Conv2D(squeeze_filters, (1, 1), name=f'fire{fire_id}/squeeze1x1', **conv_params)(x)
        x = Activation('relu', name=f'fire{fire_id}/relu_squeeze1x1')(x)

        left = Conv2D(expand_filtters, (1, 1), name=f'fire{fire_id}/expand1x1', **conv_params)(x)
        left = Activation('relu', name=f'fire{fire_id}/relu_expand1x1')(left)

        right = Conv2D(expand_filtters, (3, 3), padding='same', name=f'fire{fire_id}/expand3x3', **conv_params)(x)
        right = Activation('relu', name=f'fire{fire_id}/relu_expand3x3')(right)

        x = Concatenate(name=f'fire{fire_id}/concat')([left, right])
        return x

    return _module


def SqueezeNet(input_shape: tuple = (227, 227, 3), classes: int = 1000, **conv_params) -> Model:
    """
    Build a SqueezeNet model
    :param input_shape: shape of each input image. e.g. (227, 227, 3)
    :param classes: the number of classes to classify
    :return: SqueezeNet model
    """
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = _build_fire_module(2, 16, 64, **conv_params)(x)
    x = _build_fire_module(3, 16, 64, **conv_params)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool3')(x)

    x = _build_fire_module(4, 32, 128, **conv_params)(x)
    x = _build_fire_module(5, 32, 128, **conv_params)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool5')(x)

    x = _build_fire_module(6, 48, 192, **conv_params)(x)
    x = _build_fire_module(7, 48, 192, **conv_params)(x)
    x = _build_fire_module(8, 64, 256, **conv_params)(x)
    x = _build_fire_module(9, 64, 256, **conv_params)(x)
    x = Dropout(0.5, name='dropout9')(x)

    x = Conv2D(classes, (1, 1), name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D(name='pool10')(x)

    x = Dense(classes, activation='softmax', name='prob')(x)
    return Model(img_input, x, name='squeeze_v1.1')
