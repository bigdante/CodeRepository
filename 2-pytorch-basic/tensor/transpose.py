import torch

# 在使用transpose()进行转置操作时，pytorch并不会创建新的、转置后的tensor，
# 而是修改了tensor中的一些属性（也就是元数据），使得此时的offset和stride是与转置tensor相对应的。
# 转置的tensor和原tensor的内存是共享的！
x = torch.randn(3, 2)
y = torch.transpose(x, 0, 1)
print("修改前：")
print("x-", x)
print("y-", y)

print("\n修改后：")
y[0, 0] = 11
print("x-", x)
print("y-", y)

# 可以看到，改变了y的元素的值的同时，x的元素的值也发生了变化。
# 因此可以说，x是contiguous的，但y不是（因为内部数据不是通常的布局方式）。注意不要被contiguous的字面意思“连续的”误解，tensor中数据还是在内存中一块区域里，只是布局的问题！
# 为什么这么说：因为，y里面数据布局的方式和从头开始创建一个常规的tensor布局的方式是不一样的。这个可能只是python中之前常用的浅拷贝，y还是指向x变量所处的位置，只是说记录了transpose这个变化的布局。

# 使用contiguous()
# 如果想要断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()，针对x进行变化，感觉上就是我们认为的深拷贝。
# 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
# 代码示例：

x = torch.randn(3, 2)
y = torch.transpose(x, 0, 1).contiguous()
print("修改前：")
print("x-", x)
print("y-", y)

print("\n修改后：")
y[0, 0] = 11
print("x-", x)
print("y-", y)
