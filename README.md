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
x=dfm.call(x)
x = tf.keras.layers.Permute(dims=(2,1,3))(x)
x=dfm.call(x)
x = tf.keras.layers.Permute(dims=(2,1,3))(x)
```

# Retinal Diesease Classification
You can load ClassModelV7-0 using Keras to classify normalized 224x224 retinal images into normal and diseased classes.
```
confusion matrix
[[104  30]
 [ 13 493]]
classification report
              precision    recall  f1-score   support

           0       0.89      0.78      0.83       134
           1       0.94      0.97      0.96       506

    accuracy                           0.93       640
   macro avg       0.92      0.88      0.89       640
weighted avg       0.93      0.93      0.93       640
```
