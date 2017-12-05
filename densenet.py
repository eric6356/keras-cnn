from keras import Model, Input
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.regularizers import l2


def DenseNet(input_shape: tuple = (176, 176, 3), classes: int = 1000, **kwargs) -> Model:
    weight_decay = kwargs.get('weight_decay', 1e-4)

    img_input = Input(shape=input_shape)

    x = img_input


    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes,
              'softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    return Model(img_input, x, name='densenet')
