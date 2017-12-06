"""
DenseNet implementation in keras
https://arxiv.org/abs/1512.03385
https://arxiv.org/abs/1603.05027
"""
from keras import Input, Model
from keras.layers import Activation, MaxPooling2D, Flatten, Add, AveragePooling2D, K

from .net import BaseNet


class ResNet(BaseNet):
    def __init__(self, input_shape: tuple = (224, 224, 3), classes: int = 80, **kwargs):
        """
        ResNet builder
        :param input_shape: input shape
        :param classes: number of output classes
        """
        self.bottleneck = None
        super().__init__(input_shape, classes, **kwargs)

    def apply_residual_stage(self, x, stage: int, nb_blocks: int):
        """
        apply a residual stage to input tensor x
        :param x: input tensor
        :param stage: stage number, 2 ~ 5, int
        :param nb_blocks: number of residual blocks in this stage
        :return: output tensor
        """
        nb_filters = {2: 64, 3: 128, 4: 256, 5: 512}[stage]

        for i in range(nb_blocks):
            short_conv = i == 0
            strides = 2 if stage > 2 and short_conv else 1
            x = self.apply_residual_block(x, nb_filters, i == 0, strides)
        return x

    def apply_residual_block(self, block_input, nb_filters: int, shortcut_conv: bool, strides: int):
        """
        apply a residual unit to input tensor
        :param block_input: input tensor
        :param nb_filters: number of filters in the first conv layer
        :param shortcut_conv: whether to apply conv to the shortcut.
            typically only the first block in a stage needs this
        :param strides: strides of shortcut conv (if exists) and the first conv layer
            starts from state 3, strides = 2
        :return: output tensor
        """
        if self.bottleneck:
            bn1 = self.BatchNormalization()(block_input)
            relu1 = Activation('relu')(bn1)
            conv1 = self.Conv2D(nb_filters, 1, padding='valid')(relu1)

            bn2 = self.BatchNormalization()(conv1)
            relu2 = Activation('relu')(bn2)
            conv2 = self.Conv2D(nb_filters, 3, strides=strides, padding='same')(relu2)

            bn3 = self.BatchNormalization()(conv2)
            relu3 = Activation('relu')(bn3)
            conv3 = self.Conv2D(nb_filters * 4, 1, padding='valid')(relu3)

            residual = conv3
        else:
            bn1 = self.BatchNormalization()(block_input)
            relu1 = Activation('relu')(bn1)
            conv1 = self.Conv2D(nb_filters, 3, strides=strides, padding='same')(relu1)

            bn2 = self.BatchNormalization()(conv1)
            relu2 = Activation('relu')(bn2)
            conv2 = self.Conv2D(nb_filters, 3, padding='same')(relu2)

            residual = conv2

        if shortcut_conv:
            filters = nb_filters * 4 if self.bottleneck else nb_filters
            shortcut = self.Conv2D(filters, 1, strides=strides, padding='valid')(relu1)
        else:
            shortcut = block_input

        return Add()([shortcut, residual])

    def build(self, block_config: tuple, bottleneck: bool, name: str = 'ResNet'):
        """
        build a ResNet model
        :param block_config: number blocks in stage 2 ~ 5
        :param bottleneck: whether to use a bottleneck residual block
        :param name: model name
        :return: model
        """
        self.bottleneck = bottleneck

        img_input = Input(shape=self.input_shape)

        # Stage 1
        x = self.Conv2D(64, kernel_size=7, strides=2, padding='same')(img_input)
        x = self.BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)

        # Stage 2 ~ 5
        x = self.apply_residual_stage(x, 2, block_config[0])
        x = self.apply_residual_stage(x, 3, block_config[1])
        x = self.apply_residual_stage(x, 4, block_config[2])
        x = self.apply_residual_stage(x, 5, block_config[3])

        x = self.BatchNormalization()(x)
        x = Activation('relu')(x)

        shape = K.int_shape(x)
        x = AveragePooling2D(pool_size=(shape[1], shape[2]), strides=1)(x)

        x = Flatten()(x)
        x = self.Dense(self.classes, 'softmax')(x)

        return Model(inputs=img_input, outputs=x, name=name)

    def build18(self):
        """
        build a resnet-18 model
        :return: model
        """
        return self.build((2, 2, 2, 2), bottleneck=False, name='ResNet18')

    def build34(self):
        """
        build a resnet-34 model
        :return:
        """
        return self.build((3, 4, 6, 3), bottleneck=False, name='ResNet34')

    def build50(self):
        """
        build a resnet-50 model
        :return:
        """
        return self.build((3, 4, 6, 3), bottleneck=True, name='ResNet50')

    def build101(self):
        """
        build a resnet-101 model
        :return:
        """
        return self.build((3, 4, 23, 3), bottleneck=True, name='ResNet101')

    def build152(self):
        """
        build a resnet-152 model
        :return:
        """
        return self.build((3, 8, 36, 3), bottleneck=True, name='ResNet152')
