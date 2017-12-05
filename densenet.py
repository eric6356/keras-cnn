"""
DenseNet implementation in keras
https://github.com/liuzhuang13/DenseNetCaffe
"""

from typing import Union, Callable

from keras import Model, Input
from keras.layers import Dense, GlobalAveragePooling2D, Activation, BatchNormalization, Conv2D, Dropout, Concatenate, \
    AveragePooling2D
from keras.regularizers import l2


class DenseNet(object):
    def __init__(self,
                 input_shape: tuple = (176, 176, 3),
                 classes: int = 80,
                 **kwargs):
        """
        DenseNet builder
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

        self.layers_per_block = self.nb_blocks = self.nb_channels = self.growth_rate = None

    def Dense(self, units: int, activation: Union[str, Callable]):
        """
        wrap keras Dense
        :param units: number of output size
        :param activation: activation function, either a string or a function
        :return: Dense layer
        """
        def _Dense(x):
            return Dense(units,
                         activation=activation,
                         kernel_regularizer=l2(self.weight_decay),
                         bias_regularizer=l2(self.weight_decay))(x)

        return _Dense

    def BatchNormalization(self):
        """
        wrap keras BatchNormalization
        :return: BatchNormalization layer
        """
        def _BatchNormalization(x):
            return BatchNormalization(
                gamma_regularizer=l2(self.weight_decay),
                beta_regularizer=l2(self.weight_decay))(x)

        return _BatchNormalization

    def Conv2D(self, filters: int, kernel_size: tuple):
        """
        wrap keras Conv2D
        :param filters: number of filters
        :param kernel_size: kernel size
        :return: Conv2D layer
        """
        def _Conv2D(x):
            return Conv2D(filters,
                          kernel_size,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=l2(self.weight_decay),
                          padding=self.padding,
                          use_bias=self.use_bias)(x)

        return _Conv2D

    def apply_bn_relu_conv(self, x, nb_filters: int, kernel_size: tuple):
        """
        apply a BN -> ReLU -> CONV block to tensor x
        Original implementation: BN -> Scale -> ReLU -> CONV
        In keras, scale is actually automatically done by ReLU layer so it can be skipped
        :param x: input tensor
        :param nb_filters: number of filters in conv layer
        :param kernel_size: kernel size of conv layer
        :return: output tensor
        """
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self.Conv2D(nb_filters, kernel_size)(x)
        if self.dropout_rate:
            x = Dropout(self.dropout_rate)(x)
        return x

    def apply_dense_block(self, x):
        """
        apply a dense block to input tensor x
        :param x: input tensor
        :return: output tensor
        """
        for _ in range(self.layers_per_block):
            conv = self.apply_bn_relu_conv(x, self.growth_rate, (3, 3))
            x = Concatenate()([x, conv])
            self.nb_channels += self.growth_rate
        return x

    def apply_transition(self, x):
        """
        apply a transition layer to input tensor x
        :param x: input tensor
        :return: output tensor
        """
        x = self.apply_bn_relu_conv(x, self.nb_channels, (1, 1))
        x = AveragePooling2D()(x)
        return x

    def apply_densenet(self,
                       img_input,
                       depth: int = 40,
                       growth_rate: int = 12,
                       nb_blocks: int = 3,
                       nb_first_output: int = 16) -> Model:
        """
        apply a densenet to input tensor img_input
        :param img_input: input tensor
        :param depth: total depth of the densenet
        :param growth_rate: growth rate (k)
        :param nb_blocks: number of dense blocks in the network
        :param nb_first_output: output size of the first conv layer
        :return: output tensor
        """
        if (depth - 4) % nb_blocks:
            raise ValueError(f'depth must be {nb_blocks} N + 4')

        self.layers_per_block = int((depth - 4) / nb_blocks)
        self.nb_channels = nb_first_output
        self.growth_rate = growth_rate

        x = self.Conv2D(nb_first_output, (3, 3))(img_input)

        # dense blocks
        for i in range(nb_blocks - 1):
            x = self.apply_dense_block(x)
            x = self.apply_transition(x)

        # last dense block, no transition
        x = self.apply_dense_block(x)

        x = self.BatchNormalization()(x)
        x = Activation('relu')(x)

        x = GlobalAveragePooling2D()(x)
        x = self.Dense(self.classes, 'softmax')(x)

        return x

    def build40(self):
        """
        build a densenet40 model
        :return: model
        """
        img_input = Input(shape=self.input_shape)
        x = self.apply_densenet(img_input)
        return Model(inputs=img_input, outputs=x, name='DenseNet40')
