---
layout: post
title:  "some review about vision - from autonomous cars perspective"
date:   2023-10-30 11:42:45 +0800
categories: share
---

**一、编程基础**

**Python**

不可变对象与可变对象？

在python中，一切皆对象。每个对象都有id、类型、值，一旦对象被创建，其id不会再被改变，可以将id理解为对象在内存中的地址。

Python中的变量可以指向任意对象，可以将变量看成是指针，保存了所有对象的内存地址。

根据对象的值是否可以修改分为不可变对象和可变对象。

不可变对象包括：数字，字符串，tuple

可变对象包括：list，dict，set

**对于不可变对象**，如果要更新变量指向的不可变对象的值，会创建新的对象，改变变量指向的不可变对象。

```

In [1]: x = 1

In [2]: y = x

In [3]: y
Out[3]: 1

In [4]: id(x)
Out[4]: 4345434800

In [5]: id(y)
Out[5]: 4345434800

In [6]: y = 2

In [7]: id(y)
Out[7]: 4345434832

In [8]: x = 1

In [9]: id(x)
Out[9]: 4345434800

In [10]: x = 2

In [11]: id(x)
Out[11]: 4345434832

In [12]: x = 3

In [13]: id(x)
Out[13]: 4345434864
```

以上是int类型的一个实例，可以看到：

1. 更改指向不可变对象的变量的值，会在内存中创建一个新的对象，变量指向新的对象。
2. 对于不可变对象，不管几个引用指向它，内存中都只占用了一个地址，在python内部会通过引用计数来记录指向该地址的引用个数，当引用个数为0时会进行垃圾回收。

不可变对象的优点是对于相同的对象，无论多少个引用，在内存中只占用一个地址，缺点是更新需要创建新的对象，因此效率不高。

**对于可变对象**，如果要更新变量指向的可变对象的值，不会创建新的对象，会在原有对象上进行更新。

```
>>> a = [1, 2]

>>> b = a

>>> print(id(a), id(b))
1961088949320 1961088949320

>>> a.append(3)

>>> print(a, b)
[1, 2, 3] [1, 2, 3]

>>> print(id(a), id(b))
1961088949320 1961088949320

>>> a = [1, 2, 3]

>>> print(id(a), id(b))
1961088989704 1961088949320
```

以上是list类型的一个实例，可以看到：

1. 修改指向可变对象的变量，对象值的变化是在原有对象的基础上进行更新的，即变量引用的地址没有变化。
2. 对一个变量的两次赋值操作，虽值相同，但是引用的地址是不同的，也就是拥有相同值的对象，在内存中是拥有各自不同的地址的。
3. 对一个可变对象，可变指的是append、+=这种操作，赋值操作会新建一个对象。

**赋值、浅拷贝与深拷贝**？

赋值：对一个对象进行赋值操作，例如y=x，y，x两个变量指向同一个对象的地址，两个变量除了名称以外其他全部一样。

浅拷贝：对一个对象进行浅拷贝操作，例如y = x.copy()，新变量y指向的对象的地址和原始变量x指向的对象的地址完全不同，但是子对象的地址是相同的，可以理解为外新内旧。

深拷贝：对一个对象进行深拷贝操作，例如y = copy.deepcopy(x)，新变量y指向的对象的地址和原始变量x指向的对象的地址完全不同，而且子对象的地址是完全不相同的。

show you the code：

```
import copy

list1 = [1, 2, 3, ['a', 'b']] # 原始对象
list2 = list1                 # 赋值
list3 = list1.copy()          # 浅拷贝
list4 = copy.deepcopy(list1)  # 深拷贝

list1.append(4)               # 修改原始对象的值
list1[3].append('c')          # 修改原始对象中子对象的值

```

list1为原始对象，包含了四个元素，三个int类型的元素和一个list类型的元素。

list2为list1赋值的结果，list3为浅拷贝list1的结果，list4为深拷贝list1的结果。

然后我们对list1进行了修改，首先append一个int类型数据“4”，然后对list1的第四个元素做了修改，给第四个元素列表append一个新元素‘c’，然后我们依次打印list1、list2、list3、list4看看结果。

```python3
print('list1 = ', list1)     # list1 =  [1, 2, 3, ['a', 'b', 'c'], 4]
print('list2 = ', list2)     # list2 =  [1, 2, 3, ['a', 'b', 'c'], 4]
print('list3 = ', list3)     # list3 =  [1, 2, 3, ['a', 'b', 'c']]
print('list4 = ', list4)     # list4 =  [1, 2, 3, ['a', 'b']]
```

我们可以发现直接赋值来的list2和list1中的数据是完全一样的，同步进行了对list1的所有修改。

通过浅拷贝来的对象list2只是子对象的修改和list1保持了一致，对于对象本身的修改并没有生效。

通过深拷贝来的对象list3完全没有变化。

接下来我们深入看看这几个对象的id，以及对象中第四个元素的id，加深一下理解。

```python3
print('id(list1) = ', id(list1)) # id(list1) =  140388468855808
print('id(list2) = ', id(list2)) # id(list2) =  140388468855808
print('id(list3) = ', id(list3)) # id(list3) =  140388468855296
print('id(list4) = ', id(list4)) # id(list4) =  140388468855552

print('id(list1[3]) = ', id(list1)) # id(list1[3]) =  140176019622592
print('id(list2[3]) = ', id(list2)) # id(list2[3]) =  140176019622592
print('id(list3[3]) = ', id(list3)) # id(list3[3]) =  140176019622592
print('id(list4[3]) = ', id(list4)) # id(list4[3]) =  140176019972352
```

通过打印结果可以看出，list1和list2指向的对象的地址是完全相同的，list1、list3、list4指向的对象的地址不相同。list1、list2、list3中的子对象是同一对象，list4中的子对象和list1中的子对象不是同一对象。

**函数传参？**

python中唯一支持的参数传递模式是共享传参（call by sharing）。多数面向对象语言都采用这一模式，包括Ruby和Java。

共享传参指的是函数各形参获得实参中各个引用的副本，换个通俗的说法，函数内部的形参是实参的别名。

这会造成的结果就是，函数内部可能会修改作为参数传入的可变对象（但无法修改对象的标识，即不能将一个对象替换为另一个对象）。

```python3
def add(a, b):
    a += b
    return a

x = 1
y = 2
add(x, y)   # 3
print(x, y) # 1 2 

a = [1, 2]
b = [3, 4]
add(a, b)   # [1, 2, 3, 4]
print(a, b) # [1, 2, 3, 4] [3, 4]

a = (10, 20)
b = (30, 40)
add(a, b)   # (10, 20, 30, 40)
print(a, b) # (10, 20) (30, 40)
```

从上面代码结果可以看出，向函数传入list类型的可变对象，在函数内部对其进行修改，可变对象在函数外部也发生了相应的变化，int类型、tuple类型的不可变对象则不会出现这种情况。

**装饰器？**

**类和实例**

采用如下代码创建一个类：

```
class A():
	pass
```

虽然类中什么也没写，但是python会自动创建好_init__方法，用于在进行实例化时调用。

```
a = A()
```

上述代码表示a是由类A创建出来的一个实例。

定义术语如下：

创建出来的对象a叫做类A的实例；

创建对象a的动作叫做实例化；

类A中的属性叫做类属性；

对象a的属性叫做实例属性。

在python中，一切皆对象，类也是一个对象。

运行程序时，类同样会被加载到内存，在程序运行时，类对象在内存中只有一份，但是使用一个类可以创建出很多个对象实例。

**类属性**

类属性就是给类对象定义的属性，通常用来记录与这个类相关的特征，而不会用于记录具体对象的特征。通过“类名.”的方式可以访问类的属性。

举例说明：

```
class Tool(object):
	#使用赋值语句，定义类属性，记录创建工具对象的总数
	count = 0
	
	def __init__(self, name):
		self.name = name
		#每次新建一个工具对象，都将工具对象的总数+1
		Tool.count += 1
		
	#创建工具对象
	tool1 = Tool("aaa")
	tool2 = Tool("bbb")
	tool3 = Tool("ccc")
	
	print(Tool.count) #3
```

注意，其实访问类属性有两种方式：

**“类对象.类属性”**和**“实例对象.类属性（并不推荐）”**

如果“实例对象中定义的实例属性”和“类对象中定义的类属性”重名了，会优先访问实例对象中的实例属性；但是如果使用“实例对象.类属性 = 值”赋值语句，只会给实例对象额外添加一个实例属性，而不会影响到类对象中的类属性。

**实例属性**

每一个实例对象都有自己独立的内存空间，拥有各自不同的实例属性，可以通过“self.”调用自己的属性

举例：

```
print(tool1.name) #aaa
print(tool2.name) #bbb
print(tool3.name) #ccc
```

**实例方法、类方法、静态方法？**

python中有三种方法、实例方法、类方法和静态方法。

采用装饰器“@classmethod”来指定一个类中的某个方法为类方法，采用装饰器“@staticmethod”来指定一个类中的某个方法为静态方法，没有采用上述装饰器指定的类中的方法为实例方法。

举例：

我们定义一个Person类，其中包含一个初始化_init__方法，表明在进行实例化的时候，需要传入参数name，age，随后赋值给实例对象的name，age属性。

在Person类中，包含一个类方法fromBirthYear，一个静态方法isAdult和一个实例方法growUp。

```
from datetime import date

class Person:
	def __init__(self, name, age):
		self.name = name
		self.age = age
		
	@classmethod
	def fromBirthYear(cls, name, age):
		return cls(name, date.tody().year - year)
		
	@staticmethod
	def isAdult(age):
		return age > 18
	
	def growUp(self):
		self.age += 1
	
```

**类方法**

类方法中第一个传递进来的参数类型必须是类，可以用来实现工厂函数的作用，增加了一种对类进行实例化的手段。比如这里在进行实例化的时候，不单单可以采用直接对name和age进行赋值的方式实例化，而且可以采用输入name和出生年份year实现Person类的实例化。

```
person1 = Person("Maya", 15)
print(person1.name, person1.age) # Maya, 15

person2 = Person.fromBirthYear("John", 2002)
print(person2.name, person2.age) # John, 20( = 2022 - 2002)
```

此外，classmethod还可以用来读取或者修改工厂类自身的属性。

类方法可以通过类对象或者实例对象调用，如果是使用实例对象调用类方法，最终会转化为通过类对象调用。

```
person3 = person2.fromBirthYear("John", 2002)
print(person2.name, person2.age) # John, 20( = 2022 - 2002)
```

**静态方法**

静态方法对传递进来的参数没有限制，而且也不能调用类中的属性和其他类内定义的方法。将静态方法放到类的外边定义也不会受到任何影响，使用场景比较少。

静态方法与实例方法一样，都可以通过类对象或者实例对象调用，如果是使用实例对象调用静态方法，最终也会转而通过类对象调用。

**实例方法**

实例方法传入的第一个参数是self，表示当前正在调用该方法的实例对象，所以只有当一个类被实例化后，才能调用实例方法，否则会报错。

实例方法主要作用就是用来读取，更改实例对象的属性，进行与实例对象相关的操作。

```
print(person1.age)  # 20
person1.growUp()
print(person1.age)  # 21
```

**提问**

若一个方法内部既要访问实例属性又要访问类属性，应该定义成什么方法？

答：应该定义为实例方法，因为在类方法中没办法访问实例属性，而在实例方法中可以使用“类名.”访问类属性。

**C++**

**1. 什么是智能指针？**

<memory>头文件中，分为shared_ptr和unique_ptr。unique_ptr只允许基础指针一个所有者，shared_ptr允许有多个所有者，通过计数的方式进行管理，最好是使用make_shared标准库函数。

**2. 什么是多态，虚函数，纯虚函数？虚函数是怎么实现的？virtual修饰类表示的是什么意思？**

**多态**：通过基类的指针或引用调用虚函数时，编译时并不确定执行的是基类还是派生类的虚函数；当程序运行到该语句时，如果基类的指针指向的是基类的对象，则基类的虚函数被调用，如果指向的是派生类的对象，则派生类的虚函数被调用。

**多态的作用**：增强程序的可扩充性。

**虚函数**：虚函数是声明时在函数前面加了virtual关键字的成员函数。

**虚函数如何实现**：每个包含虚函数的类都有一个虚函数表，该类的任何对象都存放着该虚函数表的指针。根据基类指针或引用指向的对象中所存放的虚函数表地址，在虚函数表中找虚函数地址，调用虚函数，从而实现多态。

**纯虚函数**：没有函数体的虚函数，写法是在函数声明后面加“=0”，不写函数体。包含纯虚函数的类叫做抽象类，抽象类不能实例化。

**3. 什么函数不能是虚函数？**

构造函数、静态函数。

**4. public、protected、private继承方式有什么区别？**

**5. 析构函数为什么要是虚析构函数？**

若不是虚析构函数，当使用基类的指针去销毁子类对象时，不会调用子类的析构函数，会导致内存泄露。

**构造函数为什么不能被定义为虚函数？**

**6. static作用**

	1. 修饰局部变量：存储在静态存储区，默认初始化为0
	1. 修饰全局变量或全局函数：只在本文件可见，在其他文件不可见
	1. 修饰类的成员变量只能在类外初始化（如int Base::N = 10），或者用const修饰static在类内初始化
	1. 修饰类的成员函数。注意不能同时用const和static修饰类的成员函数，因为对于类中const修饰的函数，编译器会默认添加一个this指针，而static函数是属于整个类的，没有this指针，两者会产生冲突

**7. const作用**

	1. 定义常变量
	1. 修饰成员函数，表示不可修改成员变量的值

**8. C和C++的区别是什么**

C++在C的基础上添加了类。C是面向过程的，C++是面向对象的。

**9. C++特性有哪些**

智能指针、lambda表达式、auto和decltype、基于范围的for循环、override和final关键字、右值引用、无序容器

**10. new和malloc的区别是什么**

	1. new操作符从自由存储区（free store）上为对象动态分配内存空间，而malloc函数从堆上动态分配内存
	1. new是运算符，malloc是库函数
	1. new返回指定类型，malloc范围void*类型，需要强制类型转换
	1. new内存分配失败时返回bad_alloc异常，malloc返回Null
	1. 是否需要指定内存大小，new不需要，malloc需要显式指定字节大小
	1. new会调用构造函数，delete会调用析构函数，malloc和free不会
	1. new可以被重载，malloc不能

**11. 有没有用过模版编程？**

功能相同而数据不同，分为函数模版和类模版

**12. 右值引用是什么**

能出现在赋值号左边的表达式称为左值，不能出现在赋值号左边的表达式称为右值。

非const的变量都是左值，函数调用的返回对象如果不是引用，则函数调用是右值。

因为大部分引用都是引用变量的，而变量是左值，所以这些引用称为左值引用。

右值引用可以是引用无名的临时变量，主要目的是提高运行效率。方式是&&，如A&&r = A()

**13. 内存分区**

	1. 栈区：函数参数和局部变量
	1. 堆区：malloc/new手动申请
	1. 全局区（或叫静态区）：全局变量、静态变量
	1. 常量存储区：这是一块比较特殊的存储区，里面存放的是常量，不允许修改
	1. 代码区：存放二进制代码

**14. vector的遍历方式**

​	访问分为下标访问，for each访问，at()访问。at()加了越界判断，效率会低一些。

**15. stl函数sort函数**

**16. 函数对象**

如果一个类将“()“运算符重载为成员函数，这个类就成为函数对象类，这个类的对象就是函数对象。

**17. stl迭代器有哪些？迭代器失效场景？vector clear()操作发生了什么，会清内存吗，迭代器有哪些？**



**多线程知识**

**1. 多线程用到了哪些东西？原子变量的原理是什么？**

原子变量，读写锁，条件变量

**2. mutex怎么用**

shared_lock, unique_lock



**二、DL基础**

**网络训练**

**BackPropagation**

首先由Gradient Descent和Backpropagation之间的联系，介绍BP的核心作用。

假设一个神经网络具有参数
$$
\theta=\left\{w_1, w_2, \ldots, b_1, b_2, \ldots\right\}
$$
Gradient Descent就是先选择一个初始的
$$
\theta_0
$$
然后计算该参数对于Loss Function的Gradient
$$
\nabla L\left(\theta^0\right)
$$
沿着梯度下降的方向将
$$
\theta^0
$$
更新为
$$
\theta^1=\theta^0-\eta \nabla L\left(\theta^0\right)
$$
持续这个过程，得到
$$
\theta^1, \theta^2, \theta^3, ...
$$
对于一个拥有上百万个参数的神经网络来说，
$$
\nabla L\left(\theta\right)
$$


是一个上百万维的Vector，难点在于如何有效率地计算这个Vector。BackPropagation本质上就是一个提升Gradient Descent效率的算法，核心在于其可以有效率地计算出这个上百万维的Vector，从而提升神经网络的训练效率。
$$
Network \space\space parameters: \space\space \theta=\left\{w_1, w_2, \cdots, b_1, b_2, \cdots\right\}\\
Starting \space Parameters: \space \theta^0 -> \theta^1->\theta^2\\
\begin{aligned}
\\
& \nabla \mathrm{L}(\theta) \\
& =\left[\begin{array}{c}
\partial \mathrm{L}(\theta) / \partial w_0 \\
\partial \mathrm{L}(\theta) / \partial w_2 \\
\vdots \\
\partial \mathrm{L}(\theta) / \partial b_1 \\
\partial \mathrm{L}(\theta) / \partial b_2 \\
\vdots
\end{array}\right]
\end{aligned}
\\
\begin{array}{ll}
\text { Compute } \nabla \mathrm{L}\left(\theta^0\right): & \theta^1=\theta^0-\eta \nabla \mathrm{L}\left(\theta^0\right) \\
\text { Compute } \nabla \mathrm{L}\left(\theta^1\right): & \theta^2=\theta^1-\eta \nabla \mathrm{L}\left(\theta^1\right)
\end{array}
\\
\vdots \\
Millions\space of \space parameters......
\\\\
To\space compute\space the\space gradients\space effiently\space, we\space use\space backpropagation.
$$
**基础知识**

理解backpropagation所需要的基础就是求导时的链式法则（chain rule）：

情况1: 
$$
\begin{aligned}
& y=g(x) \quad z=h(y) \\
& \Delta x \rightarrow \Delta y \rightarrow \Delta z \quad \frac{d z}{d x}=\frac{d z}{d y} \frac{d y}{d x}
\end{aligned}
$$
情况2:
$$
x=g(s) \quad y=h(s) \quad z=k(x, y)\\
\Delta s\rightarrow \Delta x \rightarrow \Delta z\\
\Delta s\rightarrow \Delta y \rightarrow \Delta z\\
\frac{d z}{d s}= \frac{\partial z}{\partial x} \frac{d x}{d s} + \frac{\partial z}{\partial y} \frac{d y}{d s}
$$
**Loss对参数的偏微分**

对整个neural network，我们定义了一个loss function
$$
L(\theta)=\sum_{n=1}^N l^n(\theta)
$$
他等于所有training data的loss之和。我们把training data里任意一个样本代入到neural network里面，它会output一个
$$
y^n
$$
我们把这个output跟样本点本身的label标注的target
$$
\tilde{y}^n
$$
计算loss。如果loss较大，说明output和target之间距离很远，当前参数下神经网络的loss是比较大的，反之说明当前神经网络的参数是比较好的。

累加所有training data的loss，得到total loss，这就是神经网络的loss function，用这个函数对某一个参数w做偏微分，得到梯度的方向，即
$$
\frac{\partial L(\theta)}{\partial w}=\sum_{n=1}^N \frac{\partial l^n(\theta)}{\partial w}
$$
由这个表达式可知，只需要考虑如何计算某一个training data的loss对参数w的偏微分
$$
\frac{\partial l^n(\theta)}{\partial w}
$$
再将所有training data的loss对参数的偏微分累计求和，就可以计算出total loss对参数w的偏微分。

![](./2023-10-30-some-review-about-vision/截屏2023-10-31 16.52.22.png)

先考虑一个neuron，如上图中被红色三角形圈住的neuron。假设只有两个input，通过这个neuron，我们先得到：
$$
z=b+w_1 x_1+w_2 x_2
$$
然后经过activate function得到
$$
z_1
$$
作为后续neuron的input，再经过大量的层后，得到最终的output
$$
y_1, y_2
$$
关键问题来了，如何计算
$$
\frac{\partial l}{\partial w}
$$
按照chain rule，可以把它拆分成两项
$$
\frac{\partial l}{\partial w}=\frac{\partial z}{\partial w} \frac{\partial l}{\partial z}\\
上面等式右边，\frac{\partial z}{\partial w}的计算过程，我们称之为forward\space pass，\frac{\partial l}{\partial z}的计算过程，我们称之为backward\space pass。
$$
![](./2023-10-30-some-review-about-vision/截屏2023-10-31 17.06.54.png)

**Forward pass**
$$
计算\frac{\partial z}{\partial w}的过程比较简单，考虑两种情况：\\
(1)input\space layer作为neuron的输入：w_1前面连接的是x_1, 计算微分为\frac{\partial z}{\partial w_1} = x_1;\\

w_2前面连接的是x_2, 计算微分为\frac{\partial z}{\partial w_2} = x_2。\\
(2)hidden\space layer作为neuron的输入：hidden\space layer的输出为上一层的输出z_i经过activate function后的\tilde{z}_i, \\
于是计算微分得\frac{\partial z}{\partial w} = \tilde{z}_i。\\
可以总结出规律：计算\frac{\partial z}{\partial w} 就是看w前面连接的input是什么，那微分后的值就是什么。\\
因此只要计算出neural\space network里面每一个neuron的output就可以知道任意的z对w的偏微分。
$$
![](./2023-10-30-some-review-about-vision/截屏2023-10-31 17.29.09.png)

上图中的数据是假设activate function为sigmoid function得到的。

**Backward pass**
$$
再考虑\frac{\partial l}{\partial z}这一项，它较第一项\frac{\partial z}{\partial w}更复杂些。我们将z输入activation\space function\space \sigma(z)得到a,即a = \sigma(z)。\\
接下来a会乘上某一个weight\space w_3, 再加上其他需要加的value得到z^{\prime}，它是下一个neuron\space activation\space function的input。\\
a还会乘上另一个weight\space w_4, 再加上其他需要加的value得到z^{\prime\prime}。\\
后面还会发生很多类似的事情，但先考虑这一步。采用链式法则计算微分\frac{\partial l}{\partial z}=\frac{\partial a}{\partial z} \frac{\partial l}{\partial a}，\\
其中\frac{\partial a}{\partial z}就是activation\space function的微分\sigma^{\prime}(z), 接下来的问题是如何求\frac{\partial l}{\partial a}。\\
a会影响z^{\prime}和z^{\prime\prime}，而这两项会影响到loss\space function\space l, 根据链式法则可以得到\frac{\partial l}{\partial a}=\frac{\partial z^{\prime}}{\partial a} \frac{\partial l}{\partial z^{\prime}}+\frac{\partial z^{\prime \prime}}{\partial a} \frac{\partial l}{\partial z^{\prime \prime}}，其中\\
\frac{\partial z^{\prime}}{\partial a}=w_3, \quad \frac{\partial z^{\prime \prime}}{\partial a}=w_4, 假设\frac{\partial l}{\partial z^{\prime}}和\frac{\partial l}{\partial z^{\prime\prime}}已经被通过某种方法计算出来了，那么\frac{\partial l}{\partial z}就呼之欲出了。\\
\frac{\partial l}{\partial z}=\frac{\partial a}{\partial z} \frac{\partial l}{\partial a}=\sigma^{\prime}(z)\left(w_3 \frac{\partial l}{\partial z^{\prime}}+w_4 \frac{\partial l}{\partial z^{\prime \prime}}\right)\\
So far，最后需要解决的问题是，如何计算\frac{\partial l}{\partial z^{\prime}}和\frac{\partial l}{\partial z^{\prime\prime}}这两项，还是假设有两种不同的情况：
$$
![](./2023-10-30-some-review-about-vision/截屏2023-11-01 11.05.12.png)
$$
Case\space 1:z^{\prime}与z^{\prime\prime}的下一层为输出层。\\
此时有\frac{\partial l}{\partial z^{\prime}}=\frac{\partial y_1}{\partial z^{\prime}} \frac{\partial l}{\partial y_1}， 其中\frac{\partial y_1}{\partial z^{\prime}}是output\space layer的激活函数对z^{\prime}的偏微分，\\ \frac{\partial l}{\partial y_1}是loss对y_1的偏微分，取决于loss function的定义。综合前述可以计算出\frac{\partial l}{\partial w_1}如下，\frac{\partial l}{\partial w_2}计算同理：\\
\begin{aligned}
\frac{\partial l}{\partial w_1} & =\frac{\partial z}{\partial w_1} \frac{\partial l}{\partial z}=x_1 \sigma^{\prime}(z)\left(w_3 \frac{\partial l}{\partial z^{\prime}}+w_4 \frac{\partial l}{\partial z^{\prime \prime}}\right) \\
& =x_1 \sigma^{\prime}(z)\left(w_3 \frac{\partial y_1}{\partial z^{\prime}} \frac{\partial l}{\partial y_1}+w_4 \frac{\partial y_2}{\partial z^{\prime \prime}} \frac{\partial l}{\partial y_2}\right)
\end{aligned}
$$
![](./2023-10-30-some-review-about-vision/截屏2023-11-01 11.43.40.png)
$$
Case\space 2:z^{\prime}与z^{\prime\prime}的下一层不是输出层。\\
假设现在z^{\prime}与z^{\prime\prime}的下一层不是整个network的输出层，那z^{\prime}经过红色neuron的activation\space function得到a^{\prime},\\
然后a^{\prime}分别与w_5, w_6相乘并加上一堆值分别得到z_a, z_b,如下图所示：
$$
![](./2023-10-30-some-review-about-vision/截屏2023-11-01 11.50.54.png)
$$
要想计算\frac{\partial l}{\partial z^{\prime}}=\frac{\partial a^{\prime}}{\partial z^{\prime}} \frac{\partial l}{\partial a^{\prime}}=\sigma^{\prime}\left(z^{\prime}\right)\left(\frac{\partial z_a}{\partial a^{\prime}} \frac{\partial l}{\partial z_a}+\frac{\partial z_b}{\partial a^{\prime}} \frac{\partial l}{\partial z_b}\right)，根据前面推导可知，只要能获得\frac{\partial l}{\partial z_a}和\frac{\partial l}{\partial z_b}的值即可，\\
其他值均可通过现有信息计算出来。于是我们可以归纳出，知道\frac{\partial l}{\partial z^{\prime}}和\frac{\partial l}{\partial z^{\prime\prime}}就能计算出\frac{\partial l}{\partial a},进而得出\frac{\partial l}{\partial z}，详见公式(24)。\\
那么这里，知道\frac{\partial l}{\partial z_a}和\frac{\partial l}{\partial z_b}就能计算出 \frac{\partial l}{\partial a^{\prime}}，进而得出\frac{\partial l}{\partial z^{\prime}}，……，重复这个过程，直到找到输出层，来到刚才分析的case1，\\
我们可以算出确切的值，然后再一层一层反推回去。
$$
这个方法乍一看比较繁琐，每次算一个微分的值，都要一路往后计算，一直走到network的输出层，一层层展开得到的最终数学表达式的长度会很夸张，如果network的neuron数量很多，会对应产生大量计算。

关键问题来了，我们换一个方向来看，从output layer开始算，你就会发现：
$$
计算\frac{\partial l}{\partial z}的运算量，和原来的在forward\space path中计算\frac{\partial z}{\partial w}是一样的。\\
假设现在有6个neuron，我们要计算l对这些z的偏微分，按照原来的思路：\\
我们想要知道\frac{\partial l}{\partial z_1}和\frac{\partial l}{\partial z_2},就要去计算\frac{\partial l}{\partial z_3}和\frac{\partial l}{\partial z_4}，想要知道\frac{\partial l}{\partial z_3}和\frac{\partial l}{\partial z_4},就要去计算\frac{\partial l}{\partial z_5}和\frac{\partial l}{\partial z_6}。\\
因此，如果我们是从前向后计算偏微分，就非常没有效率，造成大量的重复计算。
$$
![](./2023-10-30-some-review-about-vision/截屏2023-11-01 15.30.19.png)
$$
如果反过来先去计算\frac{\partial l}{\partial z_5}和\frac{\partial l}{\partial z_6}，整个计算过程的效率就大幅提升了，即我们先计算\frac{\partial l}{\partial z_5}和\frac{\partial l}{\partial z_6}，\\
然后就可以计算出\frac{\partial l}{\partial z_3}和\frac{\partial l}{\partial z_4},最后就可以算出\frac{\partial l}{\partial z_1}和\frac{\partial l}{\partial z_2}，而这一整个过程，就可以转化为下图：\\
即先计算出最靠近输出层的\frac{\partial l}{\partial z_5}和\frac{\partial l}{\partial z_6}，然后再把这两个偏微分的值乘上路径上的weight并加到一起，\\
再乘上\sigma^{\prime}\left(z_1\right) \text { 与 } \sigma^{\prime}\left(z_2\right) \text {, 就得到 } \frac{\partial l}{\partial z_1} \text { 和 } \frac{\partial l}{\partial z_2} \text { 这两个偏微分的值。 }\\
这样，就完成了所有的计算，该计算过程，就叫做Backward\space pass,该计算过程中会得到大量\frac{\partial l}{\partial a},\\
将其保存起来可以加速运算。
$$
![](./2023-10-30-some-review-about-vision/截屏2023-11-01 15.36.05.png)

实际上在做Backward pass的时候，就是建另外一个neural network，本来正向neural network里面的activation function都是sigmoid function，而现在计算Backward pass的时候，就是建另外一个反向的neural network，它的activation function就是一个运算放大器op-amp，每一个反向neuron的input是loss对后面一层layer的偏微分，output则是loss对这个neuron的偏微分，做Backward pass就是通过这样一个反向neural network的运算，把loss对每一个neuron的偏微分都给算出来。

注意：若是正向做Backward pass的话，实际上每次计算一个，就需要把该neuron后面所有的都给计算一遍，会造成很多不必要的重复运算，如果写成code的形式，就相当于调用了很多次重复函数；而如果是反向做Backward pass，实际上就是把这些调用函数的过程都变成调用“值”的过程，因此可以直接计算出结果，而不需要占用过多的堆栈空间。

**总结**

最后，总结一下Back propagating是怎么做的。

为了求loss function L对某一个参数w的偏微分，根据链式法则：
$$
\frac{\partial l}{\partial w}=\frac{\partial z}{\partial w} \frac{\partial l}{\partial z}
$$

$$
计算\frac{\partial z}{\partial w}的过程称为Forward\space pass，w前面连接的是啥这个值就是啥。
$$

$$
计算\frac{\partial l}{\partial z}的过程称为Backward\space pass。从输出层开始反向计算，可以理解为不改变参数 w ,\\
，新建立了一个反向的neuron\space network，每一个neuron的input是loss对后面一层layer的的偏微分的加权累加，\\
激活函数是原有激活函数对原有激活函数输入的微分值（看不懂这里的再回去看backward\space  pass的推导），\\
output就是loss对这个neuron的偏微分:\frac{\partial l}{\partial z}
$$

![](./2023-10-30-some-review-about-vision/截屏2023-11-01 16.05.56.png)

**激活函数**

**sigmoid激活函数**

函数表达式：
$$
\operatorname{sigmoid}(x)=\frac{1}{1+e^{-x}}
$$
函数求导：
$$
\operatorname{sigmoid}(x)^{\prime}=\frac{1}{1+e^{-x}}-\frac{1}{\left(1+e^{-x}\right)^2}=\operatorname{sigmoid}(x)(1-\operatorname{sigmoid}(x))
$$
优点：由于输出值限定在0到1，表示它对每个神经元的输出进行了归一化，适合用于将概率作为输出的模型。

缺点：

计算量大（在正向传播和反向传播中都包含幂运算和除法）；

sigmoid导数取值范围是[0, 0.25]，且当x过大或过小时，sigmoid函数的导数接近于0，由于神经网络反向传播时的“链式反应”，容易造成梯度消失，难以更新网络参数。例如对于一个10层的网络，根据“0.25的10次方约等于0.000000954”，第10层的误差相对第一层卷积的参数W1的梯度将是一个非常小的值，这就是所谓的“梯度消失”；

sigmoid的输出不是0均值（即zero-centered），这会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入，随着网络的加深，会改变数据的原始分布。

![](./2023-10-30-some-review-about-vision/截屏2023-11-02 10.59.26.png)

**tanh激活函数**

函数表达式：
$$
\tanh (x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$
函数求导：
$$
\tanh (x)^{\prime}=1-\left(\frac{e^x-e^{-x}}{e^x+e^{-x}}\right)^2=1-\tanh (x)^2
$$
优点：

tanh的输出范围是（-1，1），解决了sigmoid不是zero-centered输出问题；

在靠近0处的导数值较sigmoid更大，即神经网络的收敛速度相对于sigmoid更快；在一般的分类问题中，可将tanh用于隐藏层，sigmoid用于输出层。

缺点：

计算量大的问题依然存在；

tanh导数范围在(0,1)之间，相比sigmoid导数的范围(0, 0.25)，梯度消失问题会得到缓解，但仍然存在。

![](./2023-10-30-some-review-about-vision/截屏2023-11-02 11.09.34.png)

**ReLU激活函数**

函数表达式：
$$
\operatorname{Re} L U(x)=\left\{\begin{array}{l}
x, x \geq 0 \\
0, x \leq 0
\end{array}\right.
$$
函数求导：
$$
\operatorname{Re} L U(x)^{\prime}=\left\{\begin{array}{l}
1, x \geq 0 \\
0, x \leq 0
\end{array}\right.
$$
优点：

计算速度快；

ReLU是非线性函数（所谓非线性，就是一阶导数不为常数；对ReLU求导，在输入值分别为正和为负的情况下，导数是不同的，即ReLU的导数不是常数，所以ReLU是非线性的，只是不同于sigmoid和tanh，ReLU的非线性不是光滑的）；

当ReLU的输入x为负时，ReLU输出为0，提升了神经网络的稀疏性。（深度学习是根据大批量样本数据，从错综复杂的数据关系中，找到关键信息。换句话说，就是把密集矩阵转化为稀疏矩阵，去除噪音，保留数据的关键信息，这样的模型就有了鲁棒性。ReLU将x < 0的输出置为0，就是一个去噪音，使矩阵变稀疏的过程。而且在训练过程中，这种稀疏性是动态调节的，网络会自动调整稀疏比例，保证矩阵具备最优的关键特征。）

缺点：

ReLU不是zero-centered输出；

引入了神经元死亡问题，即ReLU强制将小于0的输入置为0（屏蔽该特征），导致网络的部分神经元处于无法更新的状态；

虽然采用ReLU在“链式反应”中不会出现梯度消失，但梯度下降的幅值就完全取决于权值的乘积，这样可能会出现梯度爆炸问题；

​	可以通过两种思路解决这类问题：一是控制权值大小，让权值在（0，1）范围内；二是做梯度裁剪，控制梯度下降强度，如ReLU(x) = min(6, max(0, x))。

![](./2023-10-30-some-review-about-vision/截屏2023-11-02 11.34.09.png)

**Leaky ReLU**

函数表达式：
$$
\operatorname{Leaky} \operatorname{Re} L U(x)=\left\{\begin{array}{l}
x, x \geq 0 \\
\alpha x, x \leq 0
\end{array}\right.
$$
函数求导：
$$
\text { Leaky } \operatorname{ReLU}(x)^{\prime}=\left\{\begin{array}{l}
1, x \geq 0 \\
\alpha, x \leq 0
\end{array}\right.
$$
优点：Leaky ReLU中引入了超参数阿尔法，一般设置为0.01。在反向传播过程中，对于Leaky ReLU的输入小于零的情况，也可以计算得到一个梯度（而不是像ReLU一样直接值为0），这样就避免了神经元死亡的问题。

缺点：

相较于ReLU，神经网络的稀疏性要差一些；

引入了额外的超参数。

![](./2023-10-30-some-review-about-vision/截屏2023-11-02 11.40.20.png)

**Swish/SiLU**

函数表达式：
$$
\operatorname{swish}(x)=\frac{x}{1+e^{-x}}
$$
函数求导：
$$
\operatorname{swish}(x)^{\prime}=\frac{1+e^{-x}+x e^{-x}}{\left(1+e^{-x}\right)^2}=\operatorname{swish}(x)+\operatorname{sigmoid}(x)(1-\operatorname{swish}(x))
$$
优点：

Swish是通过NAS搜索得到的，其取值范围是[-0.278，+∞]，平滑，非单调；

Swish在深层模型上的效果优于ReLU，例如，仅仅使用Swish单元替换ReLU就能把Mobile NASNetA在ImageNet上的top-1分类准确率提高0.9%，Inception-ResNet-v的分类准确率提高0.6%。

缺点：计算量大。

![](./2023-10-30-some-review-about-vision/截屏2023-11-02 11.49.34.png)

**GeLU**

函数表达式：
$$
G e L U(x)=x P(x \leq X)=x \Phi(x)=\frac{1}{2} x\left(1+\frac{2}{\sqrt{\pi}} \int_0^{\frac{x}{\sqrt{2}}} e^{-\eta^2} d \eta\right)
$$
函数求导：
$$
G e L U(x)^{\prime}=\Phi(x)+x \phi(x)
$$
优点：

受到Dropout，ReLU等机制的影响，希望将神经网络中“不重要”的激活信息置为零，我们可以理解为，对于输入的值，我们根据它的情况乘上1或0，更数学一点的描述是，对于每一个输入x，其服从于标准正态分布N(0, 1)，它会乘上一个伯努利分布
$$
\text { Bernoulli }(\Phi(x))
$$
其中
$$
\Phi(x)=P(X \leq x)
$$
随着x的降低，它被归零的概率是升高。对于ReLU来说，这个界限就是0，输入小于0就会被归零。这一类激活函数，不仅保留了概率性，同时也保留了对输入的依赖性。好了，现在我们可以看看GeLU到底长什么样了，我们经常希望神经网络具有确定性决策，这种想法催生了GeLU激活函数的诞生。这种函数的非线性希望对输入x上的随机正则化项做一个转换，听着比较费劲，具体来说可以表示为
$$
\Phi(\mathrm{x}) \times 1 \times \mathrm{x}+(1-\Phi(\mathrm{x})) \times 0 \times \mathrm{x}=\mathrm{x} \Phi(\mathrm{x})
$$
可以理解为，对于一部分
$$
\Phi(\mathrm{x})
$$
它直接乘以输入x，而对于另一部分
$$
1-\Phi(\mathrm{x})
$$
它们需要归零。不太严格的说，上面这个表达式可以按当前输入x比其他输入大多少来缩放x；

取值范围(-0.17, +∞)，平滑，非单调，似乎是NLP领域的当前最佳，尤其在transformer模型中表现最好，被GPT-2，BERT，RoBERTa，ALBERT等NLP模型所采用。

缺点：计算量大，通常采用GeLU的近似式来替代原式计算。

![](./2023-10-30-some-review-about-vision/截屏2023-11-02 14.39.10.png)

GeLU激活函数及其导数图像（和Swish非常像。。）
$$
backup:      ::::::::::::::::::::::::
\begin{aligned}
\text { depth } & d=\alpha^\phi \\
\text { width } & w=\beta^\phi \\
\text { resolution } & r=\lambda^\phi \\
\text { s.t. } & \alpha \cdot \beta^2 \cdot \lambda^2 \approx 2 \\
& \alpha \geq 1, \beta \geq 1, \lambda \geq 1
\end{aligned}

\\

B N\left(y_j\right)^{(b)}=\gamma \cdot\left(\frac{y_j^{(b)}-\mu\left(y_j\right)}{\sigma\left(y_j\right)}\right)+\beta

\\
$$


**三、感知算法**



**四、工程实践**

