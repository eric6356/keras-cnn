# keras-cnn
keras implementation of several CNN models.

## [SqueezeNet](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1)
```
@article{SqueezeNet,
    Author = {Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer},
    Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and $<$0.5MB model size},
    Journal = {arXiv:1602.07360},
    Year = {2016}
}
```

![](./assets/squeeze1.png)

![](./assets/squeeze2.png)

### Training notes
In my dataset, SqueezeNet is super sensitive to learning rate, I'm using `Adam` optimizer and `lr=0.0003` is a good point to start with.

## DenseNet(TBD)
## ResNet(TBD)
