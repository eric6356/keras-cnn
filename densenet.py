"""
DenseNet implementation in keras
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

        self.nb_channels = self.growth_rate = None

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

    def apply_dense_block(self, x, nb_layers: int):
        """
        apply a dense block to input tensor x
        :param x: input tensor
        :param nb_layers: number of layers in this block
        :return: output tensor
        """
        for _ in range(nb_layers):
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

    def build(self,
              growth_rate: int = 12,
              block_config: tuple = (12, 12, 12),
              nb_first_output: int = 16,
              name='DenseNet') -> Model:
        """
        build a densenet model
        :param growth_rate: growth rate (k)
        :param block_config: number of layers in each dense block
        :param nb_first_output: output size of the first conv layer
        :param name: model name
        :return: model
        """

        self.nb_channels = nb_first_output
        self.growth_rate = growth_rate

        img_input = Input(shape=self.input_shape)

        x = self.Conv2D(nb_first_output, (3, 3))(img_input)

        # dense blocks
        for i, nb_layer in enumerate(block_config):
            x = self.apply_dense_block(x, nb_layer)
            if i < len(block_config) - 1:  # no transition in the last dense block,
                x = self.apply_transition(x)

        x = self.BatchNormalization()(x)
        x = Activation('relu')(x)

        x = GlobalAveragePooling2D()(x)
        x = self.Dense(self.classes, 'softmax')(x)

        return Model(inputs=img_input, outputs=x, name=name)

    def build40(self):
        """
        build a densenet-40 model
        same as https://github.com/liuzhuang13/DenseNetCaffe
        :return: model
        """
        return self.build()

    def build22(self):
        """
        build a densenet-22 model
        3 dense block, 6 conv layers in each block
        :return: model
        """
        return self.build(block_config=(6, 6, 6))

    def build121(self):
        """
        build a densenet-121 model
        :return: model
        """
        return self.build(nb_first_output=64, growth_rate=32, block_config=(6, 12, 24, 16))

    def build169(self):
        """
        build a densenet-169 model
        :return: model
        """
        return self.build(nb_first_output=64, growth_rate=32, block_config=(6, 12, 32, 32))

    def build201(self):
        """
        build a densenet-201 model
        :return: model
        """
        return self.build(nb_first_output=64, growth_rate=32, block_config=(6, 12, 48, 32))

    def build161(self):
        """
        build a densenet-161 model
        :return: model
         """
        return self.build(nb_first_output=96, growth_rate=48, block_config=(6, 12, 32, 32))
