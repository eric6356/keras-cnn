from typing import Union, Callable, Tuple

from keras.layers import Dense, BatchNormalization, Conv2D, Activation, Dropout
from keras.regularizers import l2


class BaseNet(object):
    def __init__(self,
                 input_shape: tuple = (176, 176, 3),
                 classes: int = 80,
                 **kwargs):
        """
        Network builder
        :param input_shape: input shape
        :param classes: number of output classes
        :param kwargs:
            weight_decay: l2 param applied in each layer, default 0.01
            kernel_initializer: initializer applied in each layer, default 'he_uniform'
            padding: padding strategy, default 'same'
            use_bias: use bias in the conv layer, default False
            dropout: drop out after each conv layer, default 0, no dropout
        """
        self.input_shape = input_shape
        self.classes = classes

        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.kernel_initializer = kwargs.get('kernel_initializer', 'he_uniform')
        self.padding = kwargs.get('padding', 'same')
        self.use_bias = kwargs.get('use_bias', False)
        self.dropout_rate = kwargs.get('dropout', 0)

    def Dense(self, units: int, activation: Union[str, Callable], **kwargs):
        """
        wrap keras Dense
        :param units: number of output size
        :param activation: activation function, either a string or a function
        :param kwargs: extra arguments passed to Dense
        :return: Dense layer
        """

        dense_params = dict(kernel_regularizer=l2(self.weight_decay),
                            bias_regularizer=l2(self.weight_decay))
        dense_params.update(**kwargs)

        def _Dense(x):
            return Dense(units, activation=activation, **dense_params)(x)

        return _Dense

    def BatchNormalization(self, **kwargs):
        """
        wrap keras BatchNormalization
        :param kwargs: extra arguments passed to BatchNormalization
        :return: BatchNormalization layer
        """

        bn_params = dict(gamma_regularizer=l2(self.weight_decay),
                         beta_regularizer=l2(self.weight_decay))
        bn_params.update(**kwargs)

        def _BatchNormalization(x):
            return BatchNormalization(**kwargs)(x)

        return _BatchNormalization

    def Conv2D(self, filters: int, kernel_size: Union[int, Tuple[int]], **kwargs):
        """
        wrap keras Conv2D
        :param filters: number of filters
        :param kernel_size: kernel size
        :param kwargs: extra arguments passed to Conv2D
        :return: Conv2D layer
        """

        conv_params = dict(kernel_initializer=self.kernel_initializer,
                           kernel_regularizer=l2(self.weight_decay),
                           padding=self.padding,
                           use_bias=self.use_bias)
        conv_params.update(**kwargs)

        def _Conv2D(x):
            return Conv2D(filters,
                          kernel_size,
                          **conv_params)(x)

        return _Conv2D
