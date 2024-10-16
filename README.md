# ArConvNet

## Loading pretrained model
you can load ArConvNet-B0 model using the code blow:
```
from tensorflow.keras.models import load_model
model = load_model('ClassModelV7-0.h5')
```
and continue doing whatever you want to do.

## ArConv Layer
### PyTorch version
pending...
### Tensorflow
#### Keras
```
from keras import backend

iin = tf.keras.layers.Input(shape=backend.int_shape(x)[1:])
            iout = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1,ksize),
                strides=stride,
                activation=None,
                use_bias=False,
                padding="same",
                name=prefix + "depthwise-c",
                # kernel_regularizer='l1'
            )(iin)
            dfm=tf.keras.Model(iin, iout, name=prefix + "depthwise-model")
```
