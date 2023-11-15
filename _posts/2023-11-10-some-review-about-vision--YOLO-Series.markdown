---
layout: post
title:  "2023-11-10-some-review-about-vision--YOLO-Series"
date:   2023-11-10 10:40:45 +0800
categories: share
---

**文章目录**
一、开山之作：YOLOv1
	1.1 简介
	1.2 网络结构
	1.3 实现细节
	1.4 性能表现
二、更快更准：YOLOv2
	2.1 简介
	2.2 网络结构
	2.3 改进方法
	2.4 性能表现
三、巅峰之作：YOLOv3
	3.1 简介
	3.2 网络结构
	3.3 改进之处
	3.4 性能表现
四、大神接棒：YOLOv4
	4.1 简介
	4.2 网络结构
	4.3 各种Tricks总结
	4.4 改进方法
	4.5 性能表现
五、终局之战：YOLOv5
	5.1 简介
	5.2 网络结构
	5.3 改进方法
	5.4 性能表现
六、梅开二度：YOLOv8
	6.1 简介
	6.2 网络结构
	6.3 改进方法
	6.4 性能表现

这里有一篇总结A Comprehensive Review of YOLO: From YOLOv1 and Beyond [https://arxiv.org/pdf/2304.00501.pdf]

**一、开山之作：YOLOv1**

**1.1 简介**

在YOLOv1提出之前，R-CNN系列算法在目标检测领域独占鳌头。R- CNN系列检测精度高，但由于其网络结构是双阶段（two- stage）的特点，使得它的检测速度不能满足实时性要求，饱受诟病。为了打破这一僵局，设计一种速度更快的目标检测器是大势所趋。

2016年，Joseph Redmon、Santosh Divvala、Ross Girshick等人提出了一种单阶段（one-stage）的目标检测网络。它的检测速度非常快，每秒可以处理45帧图片，能够轻松地实时运行。由于其速度之快和其使用的特殊方法，作者将其取名为：You Only Look Once（也就是我们常说的YOLO的全称），并将该成果发表在了CVPR 2016上，从而引起了广泛地关注。
YOLO的核心思想就是把目标检测转变成一个回归问题，利用整张图作为网络的输入，仅仅经过一个神经网络，得到bounding box的位置及其所属类别。

**1.2 网络结构**

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-10 10.58.34.png)

现在看来，YOLO v1的网络结构非常清晰，是一种传统的one- stage卷积神经网络：

网络输入：448 x 448 x 3的彩色图片。

中间层：由若干卷积层和最大池化层组成，用于提取图片的抽象特征。

全连接层：由两个全连接层组成，用来预测目标的位置和类别概率值。

网络输出：7 x 7 x 30的预测结果。

**1.3 实现细节**

（1）检测策略

YOLOv1采用的是“分而治之”的策略，将一张图片平均分成7 x 7的网格，每个网格分别负责预测中心点落在该网格内的目标，回忆一下，在Faster R-CNN中，是通过一个RPN来获得目标的感兴趣区域，这种方法精度高，但需要再额外训练一个RPN网络，这无疑增加了训练的负担。在YOLO v1中通过划分得到了7 x 7个网格，这49个区域就相当于是目标的感兴趣区域。通过这种方式，我们就不需要再额外设计一个RPN网络，这正是YOLO v1作为单阶段网络的简单快捷之处！

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-10 11.14.11.png)

具体实现过程：

1. 将一幅图像分成S x S个网格（grid cell），如果某个object的中心落在网格中，则这个网格就负责预测这个object。
2. 每个网格要预测B个bounding box，每个bounding box要预测（x, y, w, h）和confidence共5个值。
3. 每个网格还要预测一个类别信息，记为C个类。
4. 总的来说，S x S个网格，每个网格要预测B个bounding box，还要预测C个类。网络输出就是一个维度为S x S x (5 x B + C)的张量。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-10 11.21.30.png)

在实际过程中，YOLOv1把一张图片划分为了7×7个网格，并且每个网格预测2个Box（Box1和Box2），20个类别。所以实际上，S=7，B=2，C=20。那么网络输出的shape也就是：7×7×30。

（2）目标损失函数

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-10 11.23.08.png)

损失由三部分组成，分别是：坐标预测损失，置信度预测损失，类别预测损失。

使用的是差方和误差。需要注意的是，w和h在进行误差计算的时候取的是它们的平方根 ，原因是对不同大小的bounding box预测时，相比于大bounding box预测偏一点，小box预测偏一点算法更不能忍受。而*差方和误差函数中对同样的偏移loss是一样的*。为了缓和这一问题，作者用了一个比较取巧的办法，就是将bounding box的w和h取平方根代替原本的w和h。

定位误差比分类误差更大，所以增加对定位误差的惩罚，使
$$
\lambda_{noobj} = 0.5
$$
在每张图像中，许多网格单元不包含任何目标。训练时就会把这些网格里的框的“置信度”分数推到零，这往往超过了包含目标的框的梯度，从而可能导致模型不稳定，训练早期发散。因此要减少不包含目标的框的置信度预测的损失，使
$$
\lambda_{noobj} = 0.5
$$
**1.4 性能表现**

（1）优点

YOLO检测速度非常快。标准版本的YOLO可以每秒处理 45 张图像；YOLO的极速版本每秒可以处理150帧图像。这就意味着 YOLO 可以以小于 25 毫秒延迟，实时地处理视频。对于欠实时系统，在准确率保证的情况下，YOLO速度快于其他方法。
YOLO 实时检测的平均精度是其他实时监测系统的两倍。
迁移能力强，能运用到其他的新的领域（比如艺术品目标检测）。

（2）局限

YOLO对相互靠近的物体，以及很小的群体检测效果不好，这是因为一个网格只预测了2个框，并且都只属于同一类。
由于损失函数的问题，定位误差是影响检测效果的主要原因，尤其是大小物体的处理上，还有待加强。（因为对于小的bounding boxes，small error影响更大）
YOLO对不常见的角度的目标泛化性能偏弱。

**二、更快更准：YOLOv2**
**2.1 简介**

2017年，作者 Joseph Redmon 和 Ali Farhadi 在 YOLOv1 的基础上，进行了大量改进，提出了 YOLOv2 和 YOLO9000。重点解决YOLOv1召回率和定位精度方面的不足。

YOLOv2 是一个先进的目标检测算法，比其它的检测器检测速度更快。除此之外，该网络可以适应多种尺寸的图片输入，并且能在检测精度和速度之间进行很好的权衡。

相比于YOLOv1是利用全连接层直接预测Bounding Box的坐标，YOLOv2借鉴了Faster R-CNN的思想，引入Anchor机制。利用K-means聚类的方法在训练集中聚类计算出更好的Anchor模板，大大提高了算法的召回率。同时结合图像细粒度特征，将浅层特征与深层特征相连，有助于对小尺寸目标的检测。

YOLO9000 使用 WorldTree 来混合来自不同资源的训练数据，并使用联合优化技术同时在ImageNet和COCO数据集上进行训练，能够实时地检测超过9000种物体。由于 YOLO9000 的主要检测网络还是YOLOv2，所以这部分以讲解应用更为广泛的YOLOv2为主。

**2.2 网络结构**

YOLOv2 采用 Darknet-19 作为特征提取网络，其整体结构如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 14.50.31.png)

改进后的YOLOv2: Darknet-19，总结如下：

1. 与VGG相似，使用了很多3×3卷积核；并且每一次池化后，下一层的卷积核的通道数 = 池化输出的通道 × 2。
2. 在每一层卷积后，都增加了批量标准化（Batch Normalization）进行预处理。
3. 采用了降维的思想，把1×1的卷积置于3×3之间，用来压缩特征。
4. 在网络最后的输出增加了一个global average pooling层。
5. 整体上采用了19个卷积层，5个池化层。

为了更好的说明，这里将 Darknet-19 与 YOLOv1、VGG16网络进行对比：

VGG-16： 大多数检测网络框架都是以VGG-16作为基础特征提取器，它功能强大，准确率高，但是计算复杂度较大，所以速度会相对较慢。因此YOLOv2的网络结构将从这方面进行改进。
YOLOv1： 基于GoogLeNet的自定义网络，比VGG-16的速度快，但是精度稍不如VGG-16。
Darknet-19： 速度方面，处理一张图片仅需要55.8亿次运算，相比于VGG306.9亿次，速度快了近6倍。精度方面，在ImageNet上的测试精度为：top1准确率为72.9%，top5准确率为91.2%。

**2.3 改进方法**

（1）Batch Normalization

Batch Normalization简称BN，意思是批量标准化。2015年由 Google 研究员在论文《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》中提出。

BN 对数据进行预处理（统一格式、均衡化、去噪等）能够大大提高训练速度，提升训练效果。基于此，YOLOv2 对每一层输入的数据都进行批量标准化，这样网络就不需要每层都去学数据的分布，收敛会变得更快。

BN算法实现：
在卷积或池化之后，激活函数之前，对每个数据输出进行标准化，实现方式如下图所示：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 15.04.18.png)

前三行是对batch进行数据归一化，第四行中引入了两个附加参数，这两个参数的具体取值可以参考上面提到的 Batch Normalization 这篇论文。

（2）引入anchor box机制

在YOLO v1中，作者设计了端到端的网络，直接对边界框的位置（x, y, w, h）进行预测。这样做虽然简单，但是由于没有类似R-CNN系列的推荐区域，所以网络在前期训练时非常困难，很难收敛。于是自YOLO v2开始，引入了anchors box机制，希望通过提前筛选得到的具有代表性的先验框anchors，使得网络在训练时更加容易收敛。

在Faster R- CNN中，是通过预测bounding box与ground truth的坐标框位置的偏移量
$$
t_x, t_y
$$
间接得到bounding box位置，公式如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 15.30.34.png)

这个公式是无约束的，预测的边界框很容易向任何方向偏移。因此，每个位置预测的边界框可以落在图片任何位置，这会导致模型的不稳定。

因此 YOLOv2 在此方法上进行了一点改变：预测边界框中心点相对于该网格左上角坐标
$$
(C_x, C_y)
$$
的相对偏移量，同时为了将bounding box的中心点约束在当前网格中，使用sigmoid函数将
$$
t_x, t_y
$$
归一化处理，将值约束在0-1，这使得模型训练更稳定。

下图为 Anchor box 与 bounding box 转换示意图，其中蓝色的是要预测的bounding box，黑色虚线框是Anchor box。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 15.55.26.png)

YOLOv2在最后一个卷积层输出13x13的feature map，意味着一张图片被分成了13x13个网格。每个网格有5个anchor box来预测5个bounding box，每个bounding box预测得到5个值：
$$
t_x, t_y, t_w, t_h, t_o
$$
 最后一个参数类似YOLOv1的置信度confidence。引入anchor box机制后，通过间接预测得到的bounding box的位置的计算公式如上图中右侧表达式所示：
$$
b_x, b_y, b_w, b_h
$$
置信度t_o的计算公式为
$$
P_r(object) * IOU(b,object) = \sigma(t_o)
$$
（3）Convolution with Anchor Boxes

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 17.08.00.png)

YOLOv1 有一个致命的缺陷就是：一张图片被分成7×7的网格，一个网格只能预测一个类，当一个网格中同时出现多个类时，就无法检测出所有类。针对这个问题，YOLOv2做出了相应的改进：

首先将YOLOv1网络的FC层和最后一个Pooling层去掉，使得最后的卷积层的输出可以有更高的分辨率特征。
然后缩减网络，用416×416大小的输入代替原来的448×448，使得网络输出的特征图有奇数大小的宽和高，进而使得每个特征图在划分单元格的时候只有一个中心单元格（Center Cell）。YOLOv2通过5个Pooling层进行下采样，得到的输出是13×13的像素特征。
借鉴Faster R-CNN，YOLOv2通过引入Anchor Boxes，预测Anchor Box的偏移值与置信度，而不是直接预测坐标值。
采用Faster R-CNN中的方式，每个Cell可预测出9个Anchor Box，共13×13×9=1521个（YOLOv2确定Anchor Boxes的方法见是维度聚类，每个Cell选择5个Anchor Box）。比YOLOv1预测的98个bounding box 要多很多，因此在定位精度方面有较好的改善。

（4）聚类方法选择Anchors

Faster R-CNN中anchor box的大小和比例是按经验设定的，不具有很好的代表性。若一开始就选择了更好的、更有代表性的先验框Anchor Boxes，那么网络就更容易学到准确的预测位置了！

YOLOv2 使用 K-means 聚类方法得到 Anchor Box 的大小，选择具有代表性的尺寸的Anchor Box进行一开始的初始化。传统的K-means聚类方法使用标准的欧氏距离作为距离度量，这意味着大的box会比小的box产生更多的错误。因此这里使用其他的距离度量公式。聚类的目的是使 Anchor boxes 和临近的 ground truth boxes有更大的IOU值，因此自定义的距离度量公式为 ：
$$
d(box, centroid) = 1 - IOU(box, centroid)
$$
到聚类中心的距离越小越好，但IOU值是越大越好，所以使用1-IOU，这样就保证距离越小，IOU值越大。具体实现方法如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 17.19.54.png)

如下图所示，是论文中的聚类效果，其中紫色和灰色也是分别表示两个不同的数据集，可以看出其基本形状是类似的。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 17.23.27.png)

从下表可以看出，YOLOv2采用5种 Anchor 比 Faster R-CNN 采用9种 Anchor 得到的平均 IOU 还略高，并且当 YOLOv2 采用9种时，平均 IOU 有显著提高。说明 K-means 方法的生成的Anchor boxes 更具有代表性。为了权衡精确度和速度的开销，最终选择K=5。
（5）Fine- Grained Features

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 17.25.55.png)

细粒度特征，可理解为不同层之间的特征融合。YOLOv2通过添加一个Passthrough Layer，把高分辨率的浅层特征连接到低分辨率的深层特征（把特征堆积在不同Channel中）而后进行融合和检测。具体操作是：先获取前层的26×26的特征图，将其同最后输出的13×13的特征图进行连接，而后输入检测器进行检测（而在YOLOv1中网络的FC层起到了全局特征融合的作用），以此来提高对小目标的检测能力。

Passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上。前面的特征图维度是后面的特征图的2倍，passthrough层抽取前面层的每个2×2的局部区域，然后将其转化为channel维度，对于26×26×512的特征图，经Passthrough层处理之后就变成了13×13×2048的新特征图（特征图大小降低4倍，而channles增加4倍），这样就可以与后面的13×13×1024特征图连接在一起形成13×13×3072的特征图，然后在此特征图基础上卷积做预测。示意图如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 17.31.41.png)

**2.4 性能表现**

在VOC2007数据集上进行测试，YOLOv2在速度为67fps时，精度可以达到76.8的mAP；在速度为40fps时，精度可以达到78.6
的mAP 。可以很好的在速度和精度之间进行权衡。下图是YOLOv1在加入各种改进方法后，检测性能的改变。可见在经过多种改进方法后，YOLOv2在原基础上检测精度具有很大的提升！
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 17.34.30.png)

**三、巅峰之作：YOLOv3**
	**3.1 简介**

2018年，作者 Redmon 又在 YOLOv2 的基础上做了一些改进。特征提取部分采用darknet-53网络结构代替原来的darknet-19，利用特征金字塔网络结构实现了多尺度检测，分类方法使用逻辑回归代替了softmax，在兼顾实时性的同时保证了目标检测的准确性。

从YOLOv1到YOLOv3，每一代性能的提升都与backbone（骨干网络）的改进密切相关。在YOLOv3中，作者不仅提供了darknet-53，还提供了轻量级的tiny-darknet。如果你想检测精度与速度兼具，可以选择darknet-53作为backbone；如果你希望达到更快的检测速度，精度方面可以妥协，那么tiny-darknet是你很好的选择。总之，YOLOv3的灵活性使得它在实际工程中得到很多人的青睐！

​	**3.2 网络结构**

相比于 YOLOv2 的 骨干网络，YOLOv3 进行了较大的改进。借助残差网络的思想，YOLOv3 将原来的 darknet-19 改进为darknet-53。论文中给出的整体结构如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 17.40.33.png)

Darknet-53主要由1×1和3×3的卷积层组成，每个卷积层之后包含一个批量归一化层和一个Leaky ReLU，加入这两个部分的目的是为了防止过拟合。卷积层、批量归一化层以及Leaky ReLU共同组成Darknet-53中的基本卷积单元DBL。因为在Darknet-53中共包含53个这样的DBL，所以称其为Darknet-53。

为了更加清晰地了解darknet-53的网络结构，可以看下面这张图：
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 17.42.29.png)

为了更好的理解此图，下面我将主要单元进行说明：

​	DBL： 一个卷积层、一个批量归一化层和一个Leaky ReLU组成的基本卷积单元。
​	res unit： 输入通过两个DBL后，再与原输入进行add；这是一种常规的残差单元。残差单元的目的是为了让网络可以提取到更深层的特征，同时避免出现梯度消失或爆炸。
​	resn： 其中的n表示n个res unit；所以 resn = Zero Padding + DBL + n × res unit 。
​	concat： 将darknet-53的中间层和后面的某一层的上采样进行张量拼接，达到多尺度特征融合的目的。这与残差层的add操作是不一样的，拼接会扩充张量的维度，而add直接相加不会导致张量维度的改变。
​	Y1、Y2、Y3： 分别表示YOLOv3三种尺度的输出。
​	与darknet-19对比可知，darknet-53主要做了如下改进：

没有采用最大池化层，转而采用步长为2的卷积层进行下采样。
	为了防止过拟合，在每个卷积层之后加入了一个BN层和一个Leaky ReLU。
	引入了残差网络的思想，目的是为了让网络可以提取到更深层的特征，同时避免出现梯度消失或爆炸。
	将网络的中间层和后面某一层的上采样进行张量拼接，达到多尺度特征融合的目的。

**3.3 改进之处**

YOLOv3最大的改进之处还在于网络结构的改进，由于上面已经讲过。因此下面主要对其它改进方面进行介绍：

（1）多尺度预测
为了能够预测多尺度的目标，YOLOv3 选择了三种不同shape的Anchors，同时每种Anchors具有三种不同的尺度，一共9种不同大小的Anchors。在COCO数据集上选择的9种Anchors的尺寸如下图红色框所示：
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 18.00.07.png)

借鉴特征金字塔网的思想，YOLOv3设计了3种不同尺度的网络输出Y1、Y2、Y3，目的是预测不同尺度的目标。由于在每一个尺度网格都负责预测3个边界框，且COCO数据集有80个类。所以网络输出的张量应该是：**N ×N ×[3∗(4 + 1 + 80)]**。由下采样次数不同，得到的N不同，最终Y1、Y2、Y3的shape分别为：[13, 13, 255]、[26, 26, 255]、[52, 52, 255]。可见参见原文：
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 18.09.40.png)

（2）损失函数

对于神经网络来说，损失函数的设计也非常重要。但是YOLOv3这篇文中并没有直接给出损失函数的表达式。下面通过对源码的分析，给出YOLOv3的损失函数表达式：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-13 19.48.21.png)

对比YOLOv1中的损失函数很容易知道：位置损失部分并没有改变，仍然采用的是sum-square error的损失计算方法。但是置信度损失和类别预测均由原来的sum-square error改为了交叉熵的损失计算方法。对于类别以及置信度的预测，使用交叉熵的效果应该更好。

（3）多标签分类

YOLOv3在类别预测方面将YOLOv2的单标签分类改进为多标签分类，在网络结构中将YOLOv2中用于分类的softmax修改为逻辑分类器。在YOLOv2中，算法认定一个目标只从属于一个类别，根据网络输出类别得分的最大值，将其归为某一类。然而在一些复杂的场景中，单一目标可能从属于多个类别。

比如在一个交通场景中，某目标的种类既属于汽车也属于卡车，如果用softmax进行分类，softmax会假设这个目标只属于一个类别，这个目标只会被认定为汽车或卡车，这种分类方法就称为单标签分类。如果网络输出认定这个目标既是汽车也是卡车，这就被称为多标签分类。

为实现多标签分类就需要用逻辑分类器来对每个类别都进行二分类。逻辑分类器主要用到了sigmoid函数，它可以把输出约束在0到1，如果某一特征图的输出经过该函数处理后的值大于设定阈值，那么就认定该目标框所对应的目标属于该类。

**3.4 性能表现**

如下图所示，是各种先进的目标检测算法在COCO数据集上测试结果。很明显，在满足检测精度差不都的情况下，YOLOv3具有更快的推理速度！

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 11.41.13.png)

如下表所示，对不同的单阶段和两阶段网络进行了测试。通过对比发现，YOLOv3达到了与当前先进检测器的同样的水平。检测精度最高的是单阶段网络RetinaNet，但是YOLOv3的推理速度比RetinaNet快得多。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 11.42.21.png)

**四、大神接棒：YOLOv4**
	**4.1 简介**

2020年，YOLO系列的作者Redmon在推特上发表声明，出于道德方面的考虑，从此退出CV界。听到此消息的我，为之震惊！本以为YOLOv3已经是YOLO系列的终局之战。没想到就在今年，Alexey Bochkovskiy等人与Redmon取得联系，正式将他们的研究命名为YOLOv4。

YOLOv4对深度学习中一些常用Tricks进行了大量的测试，最终选择了这些有用的Tricks：WRC、CSP、CmBN、SAT、 Mish activation、Mosaic data augmentation、CmBN、DropBlock regularization 和 CIoU loss。

YOLOv4在传统的YOLO基础上，加入了这些实用的技巧，实现了检测速度和精度的最佳权衡。实验表明，在Tesla V100上，对MS COCO数据集的实时检测速度达到65 FPS，精度达到43.5%AP。

YOLOv4的独到之处在于：

​	是一个高效而强大的目标检测网咯。它使我们每个人都可以使用 GTX 1080Ti 或 2080Ti 的GPU来训练一个超快速和精确的目标检测器。这对于买不起高性能显卡的我们来说，简直是个福音！
​	在论文中，验证了大量先进的技巧对目标检测性能的影响，真的是非常良心!
​	对当前先进的目标检测方法进行了改进，使之更有效，并且更适合在单GPU上训练；这些改进包括CBN、PAN、SAM等。

​	**4.2 网络结构**

**最简单清晰的表示：** **YOLOv4 = CSPDarknet53（主干） + SPP附加模块（颈） + PANet路径聚合（颈） + YOLOv3（头部）**

完整的网络结构图如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 13.55.49.png)

YOLOv4的网络结构是由 CSPDarknet53、 SPP、 PANet、YOLOv3头部等组成，下面对各部分逐一讲解：

**（1）CSPDarknet53**

在YOLOv4中，将CSPDarknet53作为主干网络。在了解CSPDarknet53之前，需要先介绍下CSPNet。
CSPNet来源于这篇论文：《CSPNET： A NEW BACKBONE THAT CAN ENHANCE LEARNING CAPABILITY OF CNN》
CSPNet开源地址： https://github.com/WongKinYiu/CrossStagePartialNetworks
CSPNet全称是Cross Stage Partial Network，在2019年由Chien-Yao Wang等人提出，用来解决以往网络结构需要大量推理计算的问题。作者将问题归结于网络优化中的重复梯度信息。CSPNet在ImageNet dataset和MS COCO数据集上有很好的测试效果，同时它易于实现，在ResNet, ResNeXt和DenseNet网络结构上都能通用。

CSPNet的主要目的是能够实现更丰富的梯度组合，同时减少计算量。这个目标是通过将基本层的特征图分成两部分，然后通过一个跨阶段的层次结构合并它们来实现。

而在YOLOv4中，将原来的Darknet53结构换为了CSPDarknet53，这在原来的基础上主要进行了两项改变：

​	**将原来的Darknet53与CSPNet进行结合**。在前面的YOLOv3中，我们已经了解了Darknet53的结构，它是由一系列残差结构组成。进行结合后，CSPnet的主要工作就是将原来的残差块的堆叠进行拆分，把它拆分成左右两部分：主干部分继续堆叠原来的残差块，支路部分则相当于一个残差边，经过少量处理直接连接到最后。具体结构如下：
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 14.08.03.png)

​	**使用Mish激活函数代替了原来的Leaky ReLU**。在YOLOv3中，每个卷积层之后包含一个批量归一化层和一个Leaky ReLU。而在YOLOv4的主干网络CSPDarknet53中，使用Mish代替了原来的Leaky ReLU。Leaky ReLU和Mish激活函数的公式与图像如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 14.12.58.png)

**（2）SPP**

SPP来源于这篇论文：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://link.springer.com/content/pdf/10.1007/978-3-319-10578-9_23.pdf)

SPP最初的设计目的是用来使卷积神经网络不受固定输入尺寸的限制。在YOLOv4中，作者引入SPP，是因为它显著增加了感受野，分离出了最重要的上下文特征，并且几乎不会降低YOLOv4的运行速度。如下图所示，就是SPP中经典的空间金字塔池化层。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 14.17.17.png)

在YOLOv4中，具体的做法就是：分别利用四个不同尺度的最大池化对上层输出的feature map进行处理。最大池化的池化核大小分别为13x13, 9x9, 5x5, 1x1，其中1x1相当于不处理。

**（3）PANet**

论文链接：《Path Aggregation Network for Instance Segmentation》

这篇文章发表于CVPR2018，它提出的Path Aggregation Network (PANet)既是COCO2017实例分割比赛的冠军，也是目标检测比赛的第二名。PANet整体上可以看做是在Mask R- CNN上做多处改进，充分利用了特征融合，比如引入Bottom-up path augmentation结构，充分利用网络浅特征进行分割；引入Adaptive feature pooling使得提取到的ROI特征更加丰富；引入Fully- connected fusion，通过融合一个前景二分类支路的输出得到更加精确的分割结果。

下图是PANet的示意图，主要包含FPN、Bottom-up path augmentation、Adaptive feature pooling、Fully-connected fusion四个部分。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 15.10.34.png)

​	FPN发表于CVPR2017，主要是通过融合高低层特征提升目标检测的效果，尤其可以提高小尺寸目标的检测效果。
​	Bottom-up Path Augmentation的引入主要是考虑网络浅层特征信息对于实例分割非常重要，因为浅层特征一般是边缘形状等特征。
​	Adaptive Feature Pooling用来特征融合。也就是用每个ROI提取不同层的特征来做融合，这对于提升模型效果显然是有利无害。
​	Fully-connected Fusion是针对原有的分割支路（FCN）引入一个前背景二分类的全连接支路，通过融合这两条支路的输出得到更加精确的分割结果

在YOLOv4中，作者使用PANet代替YOLOv3中的FPN作为参数聚合的方法，针对不同的检测器级别从不同的主干层进行参数聚合。并且对原PANet方法进行了修改, 使用张量连接(concat)代替了原来的捷径连接(shortcut connection)。

**（4）YOLOv3 Head**
在YOLOv4中，继承了YOLOv3的Head进行多尺度预测，提高了对不同size目标的检测性能。YOLOv3的完整结构在上文已经详细介绍，下面我们截取了YOLOv3的Head进行分析：
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 15.13.27.png)

YOLOv4学习了YOLOv3的方式，采用三个不同层级的特征图进行融合，并且继承了YOLOv3的Head。从上图可以看出，在COCO数据集上训练时，YOLOv4的3个输出张量的shape分别是：（19，19，225）、（38，38，255）、（76，76，225）。这是因为COCO有80个类别，并且每一个网格对应3个Anchor boxes，而每个要预测的bounding box对应的5个值
$$
t_x, t_y, t_w, t_h, t_o
$$
，所以有：3 x (80+5)=255 。

**4.3 各种Tricks总结**
作者将所有的Tricks可以分为两类：

在不增加推理成本的前提下获得更好的精度，而只改变训练策略或只增加训练成本的方法，作着称之为 “免费包”（Bag of freebies）；
只增加少量推理成本但能显著提高目标检测精度的插件模块和后处理方法，称之为“特价包”（Bag of specials）
下面分别对这两类技巧进行介绍。

（1）免费包
以数据增强方法为例，虽然增加了训练时间，但不增加推理时间，并且能让模型泛化性能和鲁棒性更好。像这种不增加推理成本，还能提升模型性能的方法，作者称之为"免费包"，非常形象。下面总结了一些常用的数据增强方法：

随机缩放
翻转、旋转
图像扰动、加噪声、遮挡
改变亮度、对比对、饱和度、色调
随机裁剪（random crop）
随机擦除（random erase）
Cutout
MixUp
CutMix
常见的正则化方法有：

DropOut
DropConnect
DropBlock
平衡正负样本的方法有：

Focal loss
OHEM(在线难分样本挖掘)
除此之外，还有回归 损失方面的改进：

GIOU
DIOU
CIoU
（2）特价包
增大感受野技巧：

SPP
ASPP
RFB
注意力机制：

Squeeze-and-Excitation (SE)
Spatial Attention Module (SAM)
特征融合集成：

FPN
SFAM
ASFF
BiFPN （出自于大名鼎鼎的EfficientDet）
更好的激活函数：

ReLU
LReLU
PReLU
ReLU6
SELU
Swish
hard-Swish
后处理非极大值抑制算法：

soft-NMS
DIoU NMS

**4.4 改进方法**
除了下面已经提到的各种Tricks，为了使目标检测器更容易在单GPU上训练，作者也提出了5种改进方法：

（1）Mosaic
这是作者提出的一种新的数据增强方法，该方法借鉴了CutMix数据增强方式的思想。CutMix数据增强方式利用两张图片进行拼接，但是Mosaic使利用四张图片进行拼接。如下图所示：
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 15.15.34.png)

Mosaic数据增强方法有一个优点：拥有丰富检测目标的背景，并且在BN计算的时候一次性会处理四张图片！

（2）SAT
SAT是一种自对抗训练数据增强方法，这一种新的对抗性训练方式。在第一阶段，神经网络改变原始图像而不改变网络权值。以这种方式，神经网络对自身进行对抗性攻击，改变原始图像，以制造图像上没有所需对象的欺骗。在第二阶段，用正常的方法训练神经网络去检测目标。
（3）CmBN
CmBN的全称是Cross mini-Batch Normalization，定义为跨小批量标准化（CmBN）。CmBN 是 CBN 的改进版本，它用来收集一个batch内多个mini-batch内的统计数据。BN、CBN和CmBN之间的区别具体如下图所示：
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 15.33.59.png)

（４）修改过的SAM
作者在原SAM（Spatial Attention Module）方法上进行了修改，将SAM从空间注意修改为点注意。如下图所示，对于常规的SAM，最大值池化层和平均池化层分别作用于输入的feature map，得到两组shape相同的feature map，再将结果输入到一个卷积层，接着是一个 Sigmoid 函数来创建空间注意力。
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 16.07.45.png)

将SAM（Spatial Attention Module）应用于输入特征，能够输出精细的特征图。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 16.54.50.png)

在YOLOv4中，对原来的SAM方法进行了修改。如下图所示，修改后的SAM直接使用一个卷积层作用于输入特征，得到输出特征，然后再使用一个Sigmoid 函数来创建注意力。作者认为，采用这种方式创建的是点注意力。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 17.10.23.png)

**（５）修改过的PAN**
作者对原PAN(Path Aggregation Network)方法进行了修改, 使用张量连接(concat)代替了原来的快捷连接(shortcut connection)。如下图所示：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 17.11.21.png)
	**4.5 性能表现**

如下图所示，在COCO目标检测数据集上，对当前各种先进的目标检测器进行了测试。可以发现，YOLOv4的检测速度比EfficientDet快两倍，性能相当。同时，将YOLOv3的AP和FPS分别提高10%和12%，吊打YOLOv3!

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 17.32.42.png)

综合以上分析，总结出YOLOv4带给我们的优点有：

​	与其它先进的检测器相比，对于同样的精度，YOLOv4更快（FPS）；对于同样的速度，YOLOv4更准（AP）。
​	YOLOv4能在普通的GPU上训练和使用，比如GTX 1080Ti和GTX 2080Ti等。
​	论文中总结了各种Tricks（包括各种BoF和BoS），给我们启示，选择合适的Tricks来提高自己的检测器性能。



**五、终局之战：YOLOv5**
	**5.1 简介**

​	YOLOv5是一个在COCO数据集上预训练的物体检测架构和模型系列，它代表了Ultralytics对未来视觉AI方法的开源研究，其中包含了经过数千小时的研究和开发而形成的经验教训和最佳实践。

​	YOLOv5是YOLO系列的一个延申，您也可以看作是基于YOLOv3、YOLOv4的改进作品。YOLOv5没有相应的论文说明，但是作者在Github上积极地开放源代码，通过对源码分析，我们也能很快地了解YOLOv5的网络架构和工作原理。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 17.35.35.png)

​	**5.2 网络结构**

​	YOLOv5官方代码中，一共给出了5个版本，分别是 YOLOv5n、YOLOv5s、YOLOv5m、YOLOv5l、YOLO5x 五个模型。这些不同的变体使得YOLOv5能很好的在精度和速度中权衡，方便用户选择。

本文中，我们以较为常用的YOLOv5s进行介绍，下面是YOLOv5s的整体网络结构示意图：

![](./2023-11-10-some-review-about-vision--YOLO-Series/3811699955487_.pic.jpg)

1、input

​	和YOLOv4一样，对输入的图像进行Mosaic数据增强。Mosaic数据增强的作者也是来自Yolov5团队的成员，通过随机缩放、随机裁剪、随机排布的方式对不同图像进行拼接。

​	采用Mosaic数据增强方法，不仅使图片能丰富检测目标的背景，而且能够提高小目标的检测效果。并且在BN计算的时候一次性会处理四张图片！

2、Backbone

​	骨干网路部分主要采用的是：Focus结构、CSP结构。其中 Focus 结构在YOLOv1-YOLOv4中没有引入，作者将 Focus 结构引入了YOLOv5，用于直接处理输入的图片，**通过降维和压缩输入特征图，从而减少计算量和提高感受野，同时提高目标检测的精度和模型的表达能力**。

​	Focus重要的是切片操作，如下图所示，4x4x3的图像切片后变成2x2x12的特征图。
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-14 18.36.08.png)

以YOLOv5s的结构为例，原始608x608x3的图像输入Focus结构，采用切片操作，先变成304x304x12的特征图，再经过一次32个卷积核的卷积操作，最终变成304x304x32的特征图。

3、Neck

在网络的颈部，采用的是：FPN+PAN结构，进行丰富的特征融合，这一部分和YOLOv4的结构相同。详细内容可参考：

- [目标检测算法 YOLOv4 解析](https://blog.csdn.net/wjinjie/article/details/116793973)
- [YOLO系列算法精讲：从yolov1至yolov4的进阶之路](https://blog.csdn.net/wjinjie/article/details/107509243)

4、Head

对于网络输出，遵循YOLO系列的一贯做法，采用的是耦合的head。并且和YOLO v3，YOLO v4类似，采用了三个不同的head，进行多尺度预测，详细内容可参考：

- [目标检测算法 YOLOv4 解析](https://blog.csdn.net/wjinjie/article/details/116793973)
- [YOLO系列算法精讲：从yolov1至yolov4的进阶之路](https://blog.csdn.net/wjinjie/article/details/107509243)

**5.3 改进方法**

**1、自适应锚框计算**

​	在YOLOv3、YOLOv4中，是通过K-Means方法来获取数据集的最佳anchors，这部分操作需要在网络训练之前单独进行。为了省去这部分"额外"的操作，Yolov5的作者将此功能嵌入到整体代码中，每次训练时，自适应的计算不同训练集中的最佳锚框值。当然，如果觉得计算的锚框效果不是很好，也可以在代码中将自动计算锚框功能关闭。

```
parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
```

**2、自适应灰度填充**

为了应对输入图片尺寸 不一的问题，通常做法是将原图直接resize成统一大小，但是这样会造成目标变形，如下图所示：
![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 11.00.53.png)

为了避免这种情况的发生，YOLOv5采用了灰度填充的方式统一输入尺寸，避免了目标变形的问题。灰度填充的核心思想就是将原图的长宽等比缩放对应统一尺寸，然后对于空白部分用灰色填充。如下图所示：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 11.16.21.png)

**3、损失函数**

​	分类使用交叉熵损失函数（BCE loss），边界框回归使用CIOU loss。

​	CIOU将目标与anchor之间的中心距离、重叠率、尺度以及惩罚项都考虑进去，使得目标框回归变得更加稳定，不会像IOU和GIOU一样出现训练过程中发散等问题。而惩罚因子把**预测框长宽比**拟合**目标框的长宽比**考虑了进去。CIOU具体见：：[目标检测网络中的定位损失函数对比](https://ai-wx.blog.csdn.net/article/details/116935440)	

**5.4 性能表现**

在COCO数据集上，当输入原图的尺寸是：640x640时，YOLOv5的5个不同版本的模型的检测数据如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 11.34.02.png)

在COCO数据集上，当输入原图的尺寸是：640x640时，YOLOv5的5个不同版本的模型的检测数据如下：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 11.35.52.png)


从上表可得知，从YOLOv5n到YOLOv5x，这五个YOLOv5模型的检测精度逐渐上升，检测速度逐渐下降。根据项目要求，用户可以选择合适的模型，来实现精度与速度的最佳权衡！

**六、梅开二度：YOLOv8**
	**6.1 简介**

​	YOLOv8 与YOLOv5出自同一个团队，是一款前沿、最先进（SOTA）的模型，基于先前 YOLOv5版本的成功，引入了新功能和改进，进一步提升性能和灵活性。YOLOv8 设计快速、准确且易于使用，使其成为各种物体检测与跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 14.54.38.png)

​	**6.2 网络结构**

​	整体结构上与YOLOv5类似：CSPDarknet（Backbone主干）+ PAN- FPN（颈）+ Decoupled-Head（输出头部），但是在各模块的细节上有一些改进，并且整体上是基于anchor- free的思想，这与yolov5也有着本质上的不同。

​	![](./2023-11-10-some-review-about-vision--YOLO-Series/yolov8.png)

**6.3 改进方法**

1、Backbone

​	使用的依旧是CSP的思想，不过YOLOv5中的C3模块被替换成了C2f模块，实现了进一步的轻量化，同时YOLOv8依旧使用了YOLOv5等架构中使用的SPPF模块；

​	针对C3模块，其主要借助CSPNet提取分流的思想，同时结合残差结构的思想，设计了所谓的C3 Block，这里的CSP主分支梯度模块为BottleNeck，也就是所谓的残差模块。同时堆叠的个数由参数n来进行控制，也就是说不同规模的模型，n的值是有变化的。

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 15.22.18.png)

​	C2f模块就是参考了C3模块以及ELAN（来自YOLOv7）的思想进行的设计，让YOLOv8可以在保证轻量化的同时获得更加丰富的梯度流信息。

​	YOLOv7通过并行更多的梯度流分支，设计ELAN模块可以获得更丰富的梯度信息，进而或者更高的精度和更合理的延迟。

​	![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 15.26.12.png)

2、PAN- FPN

​	毫无疑问YOLOv8依旧使用了PAN的思想，不过通过对比YOLOv5与YOLOv8的结构图可以看到，YOLOv8将YOLOv5中PAN-FPN上采样阶段中的卷积结构删除了，同时也将C3模块替换为了C2f模块；

3、Decoupled- Head

​	与YOLOX类似，采用了解耦的输出头部，分别进行类别和边界框的回归学习。

​	![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 15.34.49.png)

4、Anchor- Free

​	YOLOv8抛弃了以往的Anchor-Base，使用了Anchor-Free的思想。

5、损失函数

​	YOLOv8使用VFL Loss作为分类损失，使用DFL Loss+CIOU Loss作为分类损失；

​	VFL主要改进是提出了非对称的加权操作，FL和QFL都是对称的。而非对称加权的思想来源于论文PISA，该论文指出首先正负样本有不平衡问题，即使在正样本中也存在不等权问题，因为mAP的计算是主正样本。

​	![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 15.41.23.png)

​	q是label，对于正样本，q为bounding box和GT的IOU，对于负样本，q=0。当为正样本的时候其实没有采用FL，而是普通的BCE，只不过多了一个自适应IoU加权，用于突出主样本。而为负样本时候就是标准的FL了。可以明显发现VFL比QFL更加简单，主要特点是正负样本非对称加权、突出正样本为主样本。

​	针对这里的DFL（Distribution Focal Loss），其主要是将框的位置建模成一个general distribution，让网络快速的聚焦于和目标位置距离近的位置的分布；DFL能够让网络更快地聚焦于目标y附近的值，增大她们的概率。

​	![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 15.54.21.png)
$$
DFL(S_i, S_{i+1}) = -((y_{i+1} - y)log(S_i) + (y - y_i)log(S_{i+1}))
$$
​	DFL的含义是以交叉熵的形式去优化与标签y最接近的一左一右2个位置的概率，从而让网络更快的聚焦到目标位置的邻近区域的分布；也就是说学出来的分布理论上是在真实浮点坐标的附近的，并且以线性插值的模式得到距离左右整数坐标的权重。

6、样本匹配

​	YOLOv8抛弃了以往的IOU匹配或者边长比例的分配方式，而是使用了Task- Aligned Assigner匹配方式。

**6.4 性能表现**

​	YOLOv8 的检测、分割和姿态模型在 COCO 数据集上进行预训练，而分类模型在 ImageNet 数据集上进行预训练。在首次使用时，模型会自动从最新的 Ultralytics 发布版本中下载；

​	YOLOv8共提供了5种不同大小的模型选择，方便开发者在性能和精度之前进行平衡。以下以YOLOv8的目标检测模型为例：	![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 16.11.40.png)

YOLOv8的分割模型也提供了5中不同大小的模型选择：

![](./2023-11-10-some-review-about-vision--YOLO-Series/截屏2023-11-15 16.13.17.png)

github: https://github.com/ultralytics/ultralytics
