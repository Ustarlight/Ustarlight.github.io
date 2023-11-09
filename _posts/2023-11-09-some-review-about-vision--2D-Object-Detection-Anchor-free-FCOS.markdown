---
layout: post
title:  "2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS"
date:   2023-11-09 10:59:45 +0800
categories: share
---

FCOS发表于ICCV 2019，是目前最常用的无需锚框的物体检测算法之一。FCOS的检测效果完全可以媲美基于锚框的单阶段法检测效果，避免了和anchor box相关的计算、超参的优化，检测流程较为简单，检测速度相对较快。核心思想是将铺设锚框变为了铺设锚点，进行物体检测。如下图所示，FCOS将原有的对锚框进行分类与回归，变为了对锚点进行分类与回归，其中回归是预测锚点到检测框上下左右四条边界的距离。

![](./2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS/截屏2023-11-09 11.17.30.png)

**算法原理与流程**

**整体框架**

FCOS与RetinaNet非常像。采用的基础网络是ResNet或ResNeXt，C3、C4、C5的特征层作为FPN的输入，输出P3、P4、P5、P6、P7作为检测层送入后续检测子网络，并在5个检测层分别对应的理论感受野中心铺设锚点，采用特定规则（后续会讲）划分锚点的正负样本。在FCOS中，5个检测层共享一个检测子网络。

在检测子网络中，分类分支和回归分支都先经过了4个卷积层进行了特征强化（与RetinaNet一样）。在分类分支中，既包含所有正、负样本锚点类别的预测分支（采用多分类Focal Loss损失函数，与RetinaNet一样），也包含一个正、负样本锚点中心性判断的分支，用来强化检测结果；回归分支用来回归正样本锚点到检测框上、下、左、右四个边界的距离（采用SmoothL1损失函数）。

![](./2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS/截屏2023-11-09 11.28.27.png)

![](./2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS/截屏2023-11-09 11.30.20.png)

### 锚点正负样本的划分

在进行锚点分类时，每一个锚点的标签是如何来的呢？

在FCOS中采用了两个比较直观上的方法来决定一个锚点是正样本还是负样本：

1. 空间限制：将位于物体真实标注框中的锚点作为候选正样本；
2. 尺度限制：FCOS为每个检测层人为设置了一个尺度范围，P3~P7检测层对应的尺度范围分别是(0, 64)、(64,128)、(128, 256)、(256, 512)、(512, +∞)，锚点回归目标（锚点到边框四条边界的距离）的最大值如果在这个范围内则是最终的正样本。这样可以使得各检测层上关联的锚点用于不同尺度物体的检测。

![](./2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS/截屏2023-11-09 11.31.53.png)

### 重叠区域的问题

多个待检测物体在一副图像中的位置发生重叠是不可避免的事情，作者发现大部分的重叠区域的GT box之间的尺度变化非常大，如下图中一个人拿着一副网球拍，大物体和小物体之间发生了重叠。上述提到的对锚点进行尺度限制很好地解决了这个问题，在检测层的浅层关联锚点进行小尺度物体的检测，在检测层的深层关联锚点进行大尺度物体的检测。如果一个锚点在同一检测层被多个物体的标注框分为正样本，作者采用最小的标注框作为回归目标。

![](./2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS/截屏2023-11-09 11.35.35.png)

### 锚点回归目标值计算

锚点的回归目标值计算公式如下图所示，其中
$$
l^*, t^*, r^*, b^*
$$
是锚点距离物体真实标注框左、上、右、下边界的距离，值得注意的是这些距离都通过锚点关联的检测层的下采样倍数S进行了归一化，使得不同尺度的物体，回归目标值都在一定范围内。

![](./2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS/截屏2023-11-09 11.41.20.png)

### 锚点的分类分支与回归分支

锚点的分类、回归任务与锚框的分类、回归任务类似，只不过正负样本的划分以及回归时采用的回归目标值有所不同（锚点的回归目标值是锚点距离标注框左、上、右、下四个边界的距离，锚框的回归目标值是锚框中心、长宽与真实标注框中心、长宽之间的偏移量）。

在FCOS中，锚点的分类采用了多分类Focal Loss作为损失函数，锚点的回归采用了SmoothL1作为损失函数。

### 中心性预测分支

FCOS中额外添加了一个中心性预测分支，其动机在于如果一个锚点距离物体标注框中心点越近，锚点的回归目标值（锚点距离标注框左、上、右、下四个边界的距离）就越一致，回归难度较低，回归效果会越好。

![](./2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS/截屏2023-11-09 13.38.57.png)

FCOS定义了如下式所示的中心性得分，距离物体中心点越近，该得分越高。当锚点位于物体标注框中心时，锚点距离左边界和右边界的距离相等，锚点距离上边界和下边界的距离相等，即
$$
l^* = r^*, t^* = b^*
$$
centerness得分为1；当锚点距离标注框中心越远时，centerness得分越小；当锚点位于标注框的角点时
$$
l^*, t^*, r^*, b^*
$$
中必有一项为0，centerness得分为0。

![](./2023-11-09-some-review-about-vision--2D-Object-Detection-Anchor-free-FCOS/截屏2023-11-09 13.43.10.png)

FCOS在训练时采用BCE LOSS对该分支进行训练（Binary Cross Entropy）。在测试时，采用该分支预测每个锚点的中心性得分，将中心性得分乘上锚点分类得分，对锚点分类得分进行加权处理，降低远离物体中心的锚点的分数，这些锚点对应的Bounding Box可能会被后续的NMS过程过滤掉，从而显著提高检测性能。

**后续改进**

1. centerness 分支的位置

原始论文：centerness 分支与 cls 分支共享前面几个卷积

改进 trick： centerness 分支挪到 reg 分支，参考 [centerness shared head · Issue #89 · tianzhi0549/FCOS](https://link.zhihu.com/?target=https%3A//github.com/tianzhi0549/FCOS/issues/89%23issuecomment-516877042)。

2. center sampling

原始论文：GT bbox 内的点，分类时均作为正样本（下图上面小图的所有黄色区域）。

改进 trick：只有 GT bbox 中心附近的 radius * stride 内的小 bbox（可以叫 center bbox）内的点，分类时才作为正样本（下图下面小图的黄色和绿色区域）。

3. reg loss

原始论文：IoU loss。

改进 trick：GIoU

4. centerness 分支的 label

原始论文：利用 l，t，r，b 计算 centerness。

改进 trick：直接用 IoU，参考 GFL 中的 Why is IoU-branch always superior than centerness-branch? [https://arxiv.org/pdf/2006.04388.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2006.04388.pdf)。小目标的 centerness 值比较小，最终 cls 分数很容易被阈值卡掉。 另外 GFL 中改进了 cls 和 reg loss。

