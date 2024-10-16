# ArConvNet

## Loading pretrained model
You can load the ArConvNet-B0 model using the code below:
```
from tensorflow.keras.models import load_model
model = load_model('ClassModelV7-0.h5')
```
After loading the model, you can continue with whatever tasks you want to perform.

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

# Retinal Diesease Classification
You can load ClassModelV7-0 using Keras to classify normalized 224x224 retinal images into normal and diseased classes.
