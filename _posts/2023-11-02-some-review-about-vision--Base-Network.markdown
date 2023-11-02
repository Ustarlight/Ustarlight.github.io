**CNN Architectures - GoogLeNet/ResNet/VGGNet**

**Introduction**

Typical CNN architectures stack a few convolutional layers(each one generally followed by a activation layers, such as relu, swish, etc.), then a pooling layer, then another few convolutional layers(+ activation layer), then another pooling layer, and so on. The image gets smaller and smaller as it progresses through the network, but it also typically gets deeper and deeper, thanks to the convolutional layers.

At the top of the stack, a regular feedforward neural network is added, composed of a few fully connected layers(+ activation later), and the final layer output the prediction(e.g., a softmax layer that outputs estimated class probabilities).

A common mistake is to use convolution kernels that are too large. For example, instead of using a convolutional layer with a 5 x 5 kernel, stack two layers with 3 x 3 kernels: **it will use fewer parameters and require fewer computations, and it will usually perform better.** One exception is for the first convolutional layer: it can typically have a large kernel(e.g., 5 x 5), usually with a stride of 2 or more: this will reduce the spatial dimension of the image without losing too much information, and since the input image has three channels in general, it will not be too costly.

 