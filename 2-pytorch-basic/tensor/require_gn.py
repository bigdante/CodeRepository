# 在学习pytorch的计算图和自动求导机制时，我们要想在心中建立一个“计算过程的图像”，需要深入了解其中的每个细节，这次主要说一下tensor的requires_grad参数。
#
# 无论如何定义计算过程、如何定义计算图，要谨记我们的核心目的是为了计算某些tensor的梯度。
# 在pytorch的计算图中，其实只有两种元素：数据（tensor）和运算，运算就是加减乘除、开方、幂指对、三角函数等可求导运算，
# 而tensor可细分为两类：叶子节点(leaf node)和非叶子节点。使用backward()函数反向传播计算tensor的梯度时，并不计算所有tensor的梯度，而是只计算满足这几个条件的tensor的梯度：
# 1.类型为叶子节点、2.requires_grad=True、3.依赖该tensor的所有tensor的requires_grad=True。
#
# 首先，叶子节点可以理解成不依赖其他tensor的tensor，在pytorch中，神经网络层中的权值w的tensor均为叶子节点；
# 自己定义的tensor例如a=torch.tensor([1.0])定义的节点是叶子节点；一个有趣的现象是：
#
import torch

a = torch.tensor([1.0])
#
print(a.is_leaf)
# True
#
b = a + 1
print(b.is_leaf)
# True
# 可以看出b竟然也是叶节点！这件事可以这样理解，单纯从数值关系上b=a+1，b确实依赖a。
# 但是从pytorch的看来，一切是为了反向求导，a的requires_grad属性为False，其不要求获得梯度，那么a这个tensor在反向传播时其实是“无意义”的，
# 可认为是游离在计算图之外的，故b仍然为叶子节点
#
#
# 使用detach()函数将某一个非叶子节点剥离成为叶子节点后，无论requires_grad属性为何值，原先的叶子节点求导通路中断，便无法获得梯度数值了。
#
# 对于需要求导的tensor，其requires_grad属性必须为True，
#
# 自己定义的tensor的requires_grad属性默认为False，神经网络层中的权值w的tensor的requires_grad属性默认为True。
# 需要说明，如果自行定义了一个tensor并将其requires_grad设置为True，该tensor是叶子节点，
# 且依赖该tensor的其他tensor是非叶子节点（非叶子节点不会自动求导），其requires_grad自动设置为True，这样便形成了一条从叶节点到loss节点的求导的“通路”。
#
# import torch
# a=torch.tensor([1.0])
# a.requires_grad=True
#
# b=a+1
# b.is_leaf
# False
# b.requires_grad
# True
# 而对于非叶子节点，其不仅requiresgrad属性为True，而且还有一个grad_fn属性，记录了该节点产生时使用的运算函数，加？乘方？使得反向求导时可以计算梯度。
# 另外，如果需要使得某一个节点成为叶子节点，只需使用detach()即可将它从创建它的计算图中分离开来。
