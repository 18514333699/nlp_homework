import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

#   读入.txt文件
loss_epoch_data = np.loadtxt("epoch_loss.txt")
#   设置x和y
x = loss_epoch_data[:, 0]
y = loss_epoch_data[:, 1]
#   创建图片
fig = plt.figure(figsize=(8, 6))  # figsize=(7,5)表示图片的大小
#   画出曲线
p = pl.plot(x, y, 'r')  # "g"代表"green",意思是曲线的颜色是绿色；”-“表示曲线是实线，label代表图例的名称
pl.legend()
pl.xlabel(u'epoch')
pl.ylabel(u'loss')
plt.show()