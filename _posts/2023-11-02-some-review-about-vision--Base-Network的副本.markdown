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



谷歌在19年提出的EfficientNet v1，从效果、参数量、速度方面均大幅超越了之前的网络。

在21年4月，谷歌团队又提出了优化版本EfficientNetV2，相比V1版本，其参数量更小，但训练速度更快。

**EfficientNet V1: Rethinking Model Scaling for Convolutional Neural Networks**

对于一个CNN网络来说，影响模型参数大小和速度的主要有三个方面：depth、width、resolution (image size)。depth指的是模型的深度，即网络的层数，越深的层感受野越大，提取的特征语义越强；width指的是网络特征维度大小（channels），特征维度越大，模型的表征能力越强；resolution指的是网络的输入图像大小，即HxW，输入图像分辨率越大，越有利于提取更细粒度特征。这三个维度，depth和width影响模型参数大小和速度，但resolution只影响模型的速度（输入图像的分辨率越大，计算量越大）。

传统的BackBone为了达到更高精度，通常采用的方式是任意增加 CNN 的深度或宽度，或使用更大的输入图像分辨率进行训练和测试， 这些方法通常需要长时间的手动调优，并且仍然会可能产生次优的性能。EfficientNet v1背后的主要设计策略是**复合缩放策略（Compound Model Scaling）**，即从**三个维度（depth，width，resolution）对模型进行缩放**，通过**神经网络架构搜索（NAS）**的方式同时对这三个维度进行平衡，**搜索得到最优网络架构**。

**复合缩放策略**

在下图中，(a)是一个baseline网络架构，(b)、(c)、(d)分别是对width、depth和resolution三个维度进行缩放，(e)是对width、depth和resolution三个维度进行复合缩放。

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.43.05.png)

Compound Model Scaling的初衷其实很好理解，相比于ResNet50到ResNet101只是增加了网络的深度depth，而WidResNet是调整原网络的width，Compound Model Scaling则是从**depth、width、resolution三个维度对模型进行调整**。论文中也通过实验发现，如果单纯地只对某一个维度scaling，随着模型增大，性能会很快达到瓶颈。 直观上看，如果模型的输入图片分辨率增加，那么应该同时增加模型深度来增大感受野以提取同样范围大小的特征，相应地，也应该增加模型width来捕获更细粒度的特征。

论文中对比了不同depth和resolution下的对width进行缩放的模型效果，如下图所示，对于depth=1.0和resolution=1.0就是只改变width，可以看到模型效果很快达到瓶颈，但是如果设置更大的depth和resolution（d=2.0和r=1.3），对width进行缩放能取得更好的效果。 

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 15.52.04.png)

那么如何设置合适的depth、width、resolution，让模型在参数量小、速度快的基础上，性能尽可能的高，是EfficientNet做出的探索（当然也是目前最主流的研究方向）。

据此，论文的compound scaling采用统一的系数phi来均衡地缩放depth，width，resolution。
$$
\begin{aligned}
\text { depth } & d=\alpha^\phi \\
\text { width } & w=\beta^\phi \\
\text { resolution } & r=\lambda^\phi \\
\text { s.t. } & \alpha \cdot \beta^2 \cdot \lambda^2 \approx 2 \\
& \alpha \geq 1, \beta \geq 1, \lambda \geq 1
\end{aligned}
$$

$$
这里的 \alpha, \beta, \lambda 是常量, 分别表示depth、width和resolution三个方面的基础系数, 这里只需要调整\phi\\
就可以实现模型的缩放。对于卷积操作, 其FLOPS一般和 d, w^2, r^2 成正比, 如果depth变为 2 倍, 那么计算量也变为2倍, \\
但是width和resolution变为2倍的话, 计算量变为 4 倍。对于baseline 模型, 其系数\phi_0=1, \\
当采用一个新的系数\phi对模型进行缩放时, 模型的FLOPS将变为baseline模型的 \left(\alpha \cdot \beta^2 \cdot \lambda^2\right)^\phi, \\
由于限制了\alpha \cdot \beta^2 \cdot \lambda^2 \approx 2, FLOPS就近似增加了 2^\phi 。
$$

**模型搜索**

论文中通过一个多目标的NAS来得到baseline模型（借鉴[MnasNet](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1807.11626)），这里优化的目标是模型的ACC和FLOPS，target FLOPS是400M，将phi固定为1，最终得到了EfficientNet-B0模型，搜索得到的最佳设置是
$$
\alpha=1.2, \beta=1.1, \lambda=1.15。
$$
EfficientNet-B0 baseline模型架构如下表所示，可以看到EfficientNet-B0的输入大小为224x224，首先是一个stride=2的3x3卷积层，最后是一个1x1卷积+global pooling+FC分类层，其余的stage主体是MBConv，这个指的是MobileNetV2中提出的mobile inverted bottleneck block（conv1x1->depthwise conv3x3->conv1x1 / +shortcut），唯一的区别是增加了SE结构来进行优化，表中的MBConv后面的数字表示的是expand_ratio（第一个1x1卷积要扩大channels的系数）。

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 16.03.41.png)

有了baseline模型，就可以按照复合缩放策略来对模型进行缩放以得到不同size的模型，即固定住
$$
\alpha, \beta, \lambda
$$
，缩放phi获得EfficientNet-B1到EfficientNet-B7。不过在谷歌开源的源码中，不同的模型是通过给出width，depth增大的系数来确定的。

```
def efficientnet_params(model_name):
  """Get efficientnet params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-b0': (1.0, 1.0, 224, 0.2),
      'efficientnet-b1': (1.0, 1.1, 240, 0.2),
      'efficientnet-b2': (1.1, 1.2, 260, 0.3),
      'efficientnet-b3': (1.2, 1.4, 300, 0.3),
      'efficientnet-b4': (1.4, 1.8, 380, 0.4),
      'efficientnet-b5': (1.6, 2.2, 456, 0.4),
      'efficientnet-b6': (1.8, 2.6, 528, 0.5),
      'efficientnet-b7': (2.0, 3.1, 600, 0.5), 
      # 后面两个是更大的模型
      'efficientnet-b8': (2.2, 3.6, 672, 0.5),
      'efficientnet-l2': (4.3, 5.3, 800, 0.5),
  }
  return params_dict[model_name]
```

EfficientNet在ImageNet数据集上的训练采用RMSProp 优化器，训练360epoch，而且策略上采用 SiLU (Swish-1) activation，AutoAugment和stochastic depth，这其实相比ResNet的训练已经增强了不少（ResNet训练epoch为90，数据增强只有随机裁剪（random-size cropping）和水平翻转（flip horizontal））。EfficientNet-B7的参数量为66M（输入大小为600），而EfficientNet-B0的参数量仅为5.3M，在ImageNet上效果如下图所示，可以看到不同size的EfficientNet在当时比其它CNN模型在acc，模型参数和FLOPS上均存在绝对性优势。

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 16.06.41.png)

然而，深度可分离卷积当年用来降低参数量和FLOPS的操作，却成了使EfficientNet变慢的重要因素，为后续EfficientNetV2埋下了伏笔。

**EfficientNet v2**

谷歌在2021年4月份提出了EfficientNet的改进版EfficientNetV2: **Smaller Models and Faster Training**。

从题目上就可以看出V2版本相比V1，模型参数量更小，训练速度更快。从下图中可以看到EfficientNetV2比其它模型训练速度快5x~11x，而参数量却少了6.8x。

EfficientNetV2指出了EfficientNet存在的三个问题：

1. **EfficientNet在非常大的图片上训练速度慢。**EfficientNet在较大的图像输入时会使用较多的显存，如果GPU/TPU总显存固定，此时就要降低训练的batch size，这会大大降低训练速度。一种解决方案就是采用较小的图像尺寸训练，但是推理时采用较大的图像尺寸。比如EfficientNet-B6的训练size采用380相比512效果还稍微更好（推理size为528），但是可以采用更大的batch size，而且计算量更小，训练速度提升2.2x。EfficientNet V2提出了一种更复杂的训练技巧——Progressive Learning解决这个问题。
2. **浅层的深度可分离卷积导致训练速度变慢。**虽然深度可分离卷积比起普通卷积有更小的参数量和FLOPS，但是深度可分离卷积需要保存的中间变量比普通卷积多，大量时间花费在读写数据上，导致训练速度变慢。EfficientNetV2通过将浅层的MBConv替换成Fused-MBConv来解决这个问题。论文发现如果将EfficientNet-B4中的MBConv替换成Fused-MBConv时，在stage1~3替换为Fused-MBConv此时训练速度提升，如果替换所有stage，虽然参数量和FLOPs大幅度提升，但是训练速度反而下降。这说明适当地组合MBConv和Fused-MBConv才能取得最佳效果，后面用NAS来进行自动搜索。
3. **每个stage的缩放系数相同是次优的。**EfficientNet的各个stage均采用一个复合缩放策略，比如depth系数为2时，各个stage的层数均加倍。但是各个stage对训练速度和参数量的影响并不是一致的，同等缩放所有stage会得到次优结果。EfficientNetV2采用的是非一致的缩放策略，后面的stages增加更多的layers。同时EfficientNet也不断地增大图像大小，由于导致较大的显存消耗而影响训练速度，所以EfficientNetV2设定了最大图像size限制为480。

**Training- aware NAS**

为了解决2和3问题，EfficientNetV2提出了Training-Aware NAS。其搜索空间和MnasNet类似，EfficientNetV2的优化目标包括accuracy，parameter efficiency和training efficienc，搜索空间为：operation types {MBConv、Fused-MBConv}，kernel size {3x3, 5x5}，expansion ratio {1, 4, 6}。其中MBConv和Fused-MBConv结构如下，Fused-MBConv将MBConv的3x3深度可分离卷积和1x1卷积合并成一个3x3的普通卷积。

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 16.51.31.png)

论文中给出了搜索得到的EfficientNetV2-S模型，结构如下所示。相比V1结构，有以下4点不同，这些区别让EfficientNetV2参数量更少，显存消耗也更少。

1. 在浅层（stage1-3）使用Fused-MBConv
2. 在浅层（stage1-4）的expansion ratio较小，而V1的各个stage的expansion ratio几乎都是6
3. V1部分stage采用了5x5卷积核，而V2只采用了3x3卷积核，但包含更多layers来弥补感受野
4. V2中也没有V1中的最后的stride-1的stage

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 16.53.14.png)

和EfficientNet类似，使用复合缩放的方法来得到EfficientNetV2-M/L。这一步还做了两点优化:

1. 将最大推理图片尺寸限制在480以下
2. 在后面的stage逐渐添加更多的层

**Progressive Learning**

除了模型设计优化，论文还提出了一种progressive learning策略来进一步提升EfficientNetV2的训练速度，简单来说就是随着训练过程进行，逐渐增加输入图像大小，同时也采用更强的正则化策略，训练的正则化策略包括数据增强和dropout等，这里更强的正则化具体指的是采用更大的dropout rate、RandAugment magnitude和mixup ratio。

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 17.00.31.png)

上图展示了作者提出的progressive learning的训练过程，即在训练的早期，使用较小的图像和弱正则化来训练网络，使网络能够轻松快速地学习简单的表示。随着图像大小的逐渐增加，也逐渐增加更强的正则化，使学习更加困难。根据输入图像大小不同采用不同的正则化策略这一操作并不难理解，因为输入图像大小越大，模型参数越多，此时应该采用更强的正则化来防止过拟合，而输入图像越小模型参数就越少，应该采用较轻的正则化策略以防止欠拟合。从下表中可以看到，大的图像输入要采用更强的数据增强，而小的图像输入要采用较轻的数据增强才能训出最优模型效果。

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 17.01.26.png)

EfficientNetV2在ImageNet上与其它模型对比如下图所示，可以看到无论是参数量，FLOPs，推理速度还是训练速度上，EfficientNetV2均有明显的优势。额外一点是，如果用ImageNet21K进行预训练，模型可以取得更好的效果，比如不用预训练，EfficientNetV2-L比EfficientNetV2-M效果仅提升了0.6个点，仅达到85.7%，但是预训练后EfficientNetV2-M效果就达到了86.2%，所以论文中说：**scaling up data size** is more effective than simply scaling up model size in high-accuracy regime。

![](./2023-11-02-some-review-about-vision--Base-Network/截屏2023-11-03 17.02.26.png)

**Tips**

深度可分离卷积为什么会导致速度变慢？

虽然深度可分离卷积比起普通卷积有更小的参数量和FLOPS，但是深度可分离卷积需要保存的中间变量比普通卷积多，大量时间花费在读写数据上，导致训练速度变慢。具体参考[FLOPs与模型推理速度](https://zhuanlan.zhihu.com/p/122943688)。

**总结**

相比于GoogleNet、ResNet这种人工设计的经典BackBone，EfficientNet系列利用强大的计算资源对网络结果进行暴力搜索，得到一系列性能、参数量、计算量最优的网络结构和一些看不懂的超参（虽然人工设计网络中的超参也是大量试出来的，可解释性也较差）。

EfficientNet系列的研究方式应该是以后发展的一个重要方向，研究人员**可以在Conv层的优化、训练策略上多下功夫研究，至于网络架构怎么组合最优，交给机器去做就好了，用魔法打败魔法**。

这项工作需要巨量的计算资源，目前确实只有寥寥几家公司可以玩的起，希望多看到一些类似的工作，并让更多玩家入局。
