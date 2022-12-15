import os.path
import random
import torch
from torch.nn import functional as F
import glob
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict


# a = torch.randn([2,3,4,5])

# print(a.shape)
# print(a)
# a = a.expand((3, *a.shape[1:]))
# print(a.shape)
# print(a)

# pad没两个一组，从前往后分别是-1维，-2维，以此类推
# 2个一组，-1维是0,0，-2维是1,2，也就是说-1维前后分别+0,-2维前后分别+1和2
# pad1 = (0, 0, 1, 2)  # 2, 3, 7, 5
# -1维前后分别+1和2，-2维前后分别+0,0
# pad2 = (1, 2, 0, 0)  # 2, 3, 7, 8
# 从左到右每两个一组，分别是-1/-2/-3/-4维前后各自加多少
# pad3 = (1, 2, 3, 4, 0, 0, 2, 7) # 11, 3, 14, 11
# a = F.pad(a, pad1)
# print(a.shape)
# a = F.pad(a, pad2)
# print(a.shape)
# a = F.pad(a, pad3)
# print(a.shape)
# a = glob.glob(r"D:\CV\Erke\yolov3_spp\*")/
# for f in a:
#     print(f)
# class A(object):
#     count = 666
#     def __init__(self):
#         self.a = 1
#     def func(self):
#         print(self.__class__.count)
#         print(self.__class__().count)

# a = A()
# print(a.count)
# a.func()
# print(a.__dict__)  # 返回实例属性
# print(a.__class__().__dict__)      # 创建一个新的实例，访问其__dict__属性，返回新实例的实例属性
# print(A.__dict__)                  # 返回类的__dict__
# print(A.__class__)                 # 总是返回<class 'type'>
# print(a.__class__.__dict__)        # 访问A类的引用的__dict__属性，同A.__dict__
# print("*" * 10)
# print(__name__)
# a = [1]
# b = [1,2]
# a.extend(b)
# a = a + b
# print(a)
# a = torch.tensor([[1],[1]])
# print(a.shape)
# a = torch.cat([a], 1)
# print(a)
# print(a.shape)
# a = torch.zeros(1)
# print(a)
# print(a.shape)
# with open("./data/my_data.data", "r") as f:
    # fd1 = f.readline()
    # fd2 = f.readlines()
    # fd3 = f.read().splitlines()

# print(f"fd2:{fd2}")
# print(f"fd3:{fd3}")
# print(f"fd1:{fd1}")
# path = r"D:\CV\Erke\data\labels\000beee2-BBC_China_8520.txt"
# aaa = str(Path(path).parent) + ".npy"
# print(aaa)
# if os.path.isfile(aaa):
#     print("1")
#     x = np.load(aaa)
#     print(len(x))
# pp = Path(path)
# print(pp)
# print(type(pp))
# a = torch.tensor((0, 5), dtype=torch.float32) * 5
# a = [np.zeros((0, 5), dtype=np.float32)] * 5
# print(a)
# b = np.array([1, 2, 3, 4, 5])
# a[0] = b
# print(a)
# for i in range(100):
#     print(random.randint(1, 11))
# img = cv2.imread(r"D:\CV\Erke\yolov3_spp\yolov3spp.png")
# print(img.shape)
# print(type(img))
# a, b = 1, 2
# a = b = 3
# print(a, b)
# def fun(x):
#     lf = lambda x: x**2
#     print(lf(x))
# fun(2)
# for i, j in zip([1, 2], [2]):
#     print(i, j)
# a = defaultdict()
# print(a)
# a[1] = 1
# a.setdefault(2, 2)
# print(a)
# print(a[1])
# fmt = "{value:.4f} ({global_avg:.4f})"
# print(fmt.format(
#             median=1,
#             global_avg=4,
#             max=1,
#             value=2)
# )
# print(len("123"))
#
# a = ".".join(["123", "456", "789"])
# print(a)
# b = "\t".join(["1232", "45", "78910", "2342", "23423423", "233", "2342"])
# print(b)
# a = "\t"  # 一个制表位，对齐每一列，每个元素向后补全一个制表位
# print(f"111{a}111")
# a = random.randrange(1, 10) * 32
# print(a)
a = torch.tensor([[1, 2, 3],[2, 3, 4]])
b = torch.tensor([[True, False, True],[True, True, True]])
print(a[b])




























