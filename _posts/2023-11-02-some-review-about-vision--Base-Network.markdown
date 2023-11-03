---
layout: post
title:  "some review about vision - from autonomous cars perspective - base network"
date:   2023-11-03 11:42:45 +0800
categories: share
---

**CNN Architectures - GoogLeNet/ResNet/VGGNet**

**Introduction**

Typical CNN architectures stack a few convolutional layers(each one generally followed by a activation layers, such as relu, swish, etc.), then a pooling layer, then another few convolutional layers(+ activation layer), then another pooling layer, and so on. The image gets smaller and smaller as it progresses through the network, but it also typically gets deeper and deeper, thanks to the convolutional layers.

At the top of the stack, a regular feedforward neural network is added, composed of a few fully connected layers(+ activation later), and the final layer output the prediction(e.g., a softmax layer that outputs estimated class probabilities).

A common mistake is to use convolution kernels that are too large. For example, instead of using a convolutional layer with a 5 x 5 kernel, stack two layers with 3 x 3 kernels: **it will use fewer parameters and require fewer computations, and it will usually perform better.** One exception is for the first convolutional layer: it can typically have a large kernel(e.g., 5 x 5), usually with a stride of 2 or more: this will reduce the spatial dimension of the image without losing too much information, and since the input image has three channels in general, it will not be too costly.

Here is how you can implement a simple CNN to tackle the Fashion MNIST dataset using keras:

```python
model = keras.models.Sequential([
     keras.layers.Conv2D(64, 7, activation="relu", padding="same",input_shape=[28, 28, 1]),
     keras.layers.MaxPooling2D(2),
     keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
     keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
     keras.layers.MaxPooling2D(2),
     keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
     keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
     keras.layers.MaxPooling2D(2),
     keras.layers.Flatten(),
     keras.layers.Dense(128, activation="relu"),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(64, activation="relu"),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(10, activation="softmax") ])
```

 Let's go through some interesting details in this model:

​	The number of filters grows as we climd up the CNN toward the output layer(it is initially 64, then 256): **it makes sense for it to grow, since the number of low-level features is often fairly low(e.g., small circles, horizontal lines), but there are many different ways to combine them into higher-level features.** It is a common practice to double the number of filters after each pooling layer: since a pooling layer divides each spatial dimension by a factor of 2, we can afford to double the number of features maps in the next layer without fear of exploding the number of parameters, memory usage, or computational load.

​	After CNN layers is the fully connected network, composed of two hidden dense layers and a dense output layer. **Note that we must flatten its inputs, since a dense network expects a 1D array of features for each instance.**

Over the years, variants of this fundamental architecture have been developed, leading to amazing advances in the field. A good measure of this progress is the error rate in competitions such as the ILSVRC ImageNet challenge. In this competition the topfive error rate for image classification fell from over 26% to less than 2.3% in just six years. The top-five error rate is the number of test images for which the system’s top five predictions did not include the correct answer. The images are large (256 pixels high) and there are 1,000 classes, some of which are really subtle (try distinguishing 120 dog breeds). Looking at the evolution of the winning entries is a good way to understand how CNNs work.

*We will first look at two of the winners of the ILSVRC challenge: GoogLeNet (2014) and ResNet (2015), then a really simple and classical architecture VGGNet (2014).*

**GoogLeNet**

The GoogLeNet architecture was developed by Christian Szegedy et al. from Google Research, and it won the ILSVRC 2014 challenge by pushing the top-five error rate below 7%. This great performance came in large part from the fact that the network was much deeper than previous CNNs. This was made possible by subnetworks called **inception modules**, which allow GoogLeNet to use parameters much more efficiently than previous architectures: GoogLeNet actually has 10 times fewer parameters than AlexNet.

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-02 16.44.18.png)

The above figure shows the architecture of an inception module. The notation “3 × 3 + 1(S)” means that the layer uses a 3 × 3 kernel, stride 1, and "same" padding. The input signal is first copied and fed to four different layers. All convolutional layers use the ReLU activation function. **Note that the second set of convolutional layers uses different kernel sizes (1 × 1, 3 × 3, and 5 × 5), allowing them to capture patterns at different scales.** Also note that every single layer uses a stride of 1 and "same" padding (even the max pooling layer), so their outputs all have the same height and width as their inputs. This makes it possible to concatenate all the outputs along the depth dimension in the final depth concatenation layer (i.e., stack the feature maps from all four top convolutional layers).

In short, you can think of the whole inception module as a convolutional layer on steroids, **able to output feature maps that capture complex patterns at various scales**.

You may wonder why inception modules have convolutional layers with 1 × 1 kernels. Surely these layers cannot capture any features because they look at only one pixel at a time? In fact, the layers serve three purposes:

​	Although they cannot capture spatial patterns, **they can capture patterns along the depth dimension.**

​	They are configured to **output fewer feature maps than their inputs**, so they serve as bottleneck layers, meaning they **reduce dimensionality**. This cuts the computational cost and the number of parameters, speeding up training and improving generalization.

​	**Each pair of convolutional layers ([1 × 1, 3 × 3] and [1 × 1, 5 × 5]) acts like a single powerful convolutional layer, capable of capturing more complex patterns**. Indeed, instead of sweeping a simple linear classifier across the image (as a single convolutional layer does), this pair of convolutional layers sweeps a two-layer neural network across the image.

Now let’s look at the architecture of the GoogLeNet CNN. The number of feature maps output by each convolutional layer and each pooling layer is shown before the kernel size. The architecture is so deep that it has to be represented in three columns, but GoogLeNet is actually one tall stack, including nine inception modules (the boxes with the spinning tops陀螺). The six numbers in the inception modules represent the number of feature maps output by each convolutional layer in the module (in the same order as in above figure). Note that all the convolutional layers use the ReLU activation function.

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 11.37.10.png)

There are several iterations of GoogLeNet.

​	Inception v2:  Add a batch normalization layer.

​	Inception v3:  Make adjustments to the inception block, and use label smoothing for model regularization.

​	Inception v4:  Include it in the residual connection.

GoogLeNet: as well as its succeeding versions, was one of the most efficient models on ImageNet, providing similar test accuracy with lower computational complexity.

**ResNet**

Kaiming He et al. won the ILSVRC 2015 challenge using a Residual Network (or ResNet), that delivered an astounding top-five error rate under 3.6%. The winning variant used an extremely deep CNN composed of 152 layers (other variants had 34, 50, and 101 layers). It confirmed the general trend: models are getting deeper and deeper, with fewer and fewer parameters. **The key to being able to train such a deep network is to use skip connections (also called shortcut connections): the signal feeding into a layer is also added to the output of a layer located a bit higher up the stack**. Let’s see why this is useful.

When training a neural network, the goal is to make it model a target function h(x). If you add the input x to the output of the network (i.e., you add a skip connection), then the network will be forced to **model f(x) = h(x) – x** rather than h(x). This is called residual learning.

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.04.32.png)

When you initialize a regular neural network, its weights are close to zero, so the network just outputs values close to zero. If you add a skip connection, the resulting network just outputs a copy of its inputs; in other words, it initially models the identity function. If the target function is fairly close to the identity function (which is often the case), this will speed up training considerably.

Moreover, if you add many skip connections, the network can start making progress even if several layers have not started learning yet. Thanks to skip connections, the signal can easily make its way across the whole network. The deep residual network can be seen as a stack of residual units (RUs), where each residual unit is a small neural network with a skip connection.

（ResNet另一个很重要的作用是可以防止梯度消失，使得梯度能够一直保持在一个足够大的值；而SGD的精髓就在于只要一直有一个足够大的梯度，总会收敛到一个比较好的地方。）

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.05.43.png)

Now let’s look at ResNet’s architecture. It is surprisingly simple. It starts and ends exactly like GoogLeNet (except without a dropout layer), and in between is just a very deep stack of simple residual units. Each residual unit is composed of two convolutional layers (and no pooling layer!), with Batch Normalization (BN) and ReLU activation, using 3 × 3 kernels and preserving spatial dimensions (stride 1, "same" padding).

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.19.54.png)

**Note that the number of feature maps is doubled every few residual units, at the same time as their height and width are halved (using a convolutional layer with stride 2)**. When this happens, **the inputs cannot be added directly to the outputs of the residual unit** because they don’t have the same shape (for example, this problem affects the skip connection represented by the dashed arrow in follow figure). To solve this problem, the inputs are passed through a 1 × 1 convolutional layer with stride 2 and the right number of output feature maps.

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.20.57.png)

ResNet-34 is the ResNet with 34 layers (only counting the convolutional layers and the fully connected layer) containing 3 residual units that output 64 feature maps, 4 RUs with 128 maps, 6 RUs with 256 maps, and 3 RUs with 512 maps.

ResNet-152 contains 3 such RUs that output 256 maps, then 8 RUs with 512 maps, a whopping 36 RUs with 1,024 maps, and finally 3 RUs with 2,048 maps.

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.22.25.png)

ResNets deeper than 34 layers use slightly different residual units. Instead of two 3 × 3 convolutional layers with, say, 256 feature maps, they use three convolutional layers: first a 1 × 1 convolutional layer with just 64 feature maps (4 times less), which acts as a bottleneck layer (as discussed already), then a 3 × 3 layer with 64 feature maps, and finally another 1 × 1 convolutional layer with 256 feature maps (4 times 64) that restores the original depth.

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.23.49.png)

**VGGNet**

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.27.16.png)

The runner-up in the ILSVRC 2014 challenge was VGGNet, developed by Karen Simonyan and Andrew Zisserman from the Visual Geometry Group (VGG) research lab at Oxford University. It had a very simple and classical architecture, with 2 or 3 convolutional layers and a pooling layer, then again 2 or 3 convolutional layers and a pooling layer, and so on (reaching a total of just 16 or 19 convolutional layers, depending on the VGG variant), plus a final dense network with 2 hidden layers and the output layer. It used only 3 × 3 filters, but many filters.

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.31.40.png)

VGGs constructs a network using reusable convolutional blocks. Different VGG models can be defined by the differences in the number of convolutional layers and output channels in each block. **The use of blocks leads to very compact representations of the network definition. It allows for efficient design of complex networks.**

In their VGG paper, Simonyan and Ziserman experimented with various architectures. In particular, they found that **several layers of deep and narrow convolutions (i.e., 3×3) were more effective than fewer layers of wider convolutions.**

