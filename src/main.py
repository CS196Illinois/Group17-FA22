import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

count = 0

img = tf.io.read_file(f"../China Dataset/Eight Immortals Crossing the Sea Scenic Area.jpeg")
img = tf.io.decode_jpeg(img)
img = tf.image.rgb_to_grayscale(img)

# convert to tensor (specify 3 channels explicitly since png files contains additional alpha channel)
# set the dtypes to align with pytorch for comparison since it will use uint8 by default

# tensor = tf.convert_to_tensor(img, dtype=tf.float32)
tensor = tf.image.convert_image_dtype(img, dtype=tf.float32)
# print(f"initial image tensor shape{tensor.shape}")

tensor = tf.image.resize(tensor, [264, 264])
# print(f"image tensor shape after reshape {tensor.shape}")

# add another dimension at the front to get NHWC shape
input_tensor = tf.expand_dims(tensor, axis=0)
# print(f"input tensor shape in NHWC format{input_tensor.shape}")

# print("image after resize")
# plt.imshow(tf.squeeze(tensor), cmap='gray')

kernel = tf.constant([
    [-1, -1, -1],
    [-1,  1, -1],
    [-1, -1, -1],
], dtype=tf.dtypes.float32)

kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

image_filter = tf.nn.conv2d(
    input=input_tensor,
    filters=kernel,
    strides=1,
    padding='SAME',
)

# print("This is our image after applying our filter. This is what our model looks at during the convolution step")
plt.imshow(tf.squeeze(image_filter))
plt.show();