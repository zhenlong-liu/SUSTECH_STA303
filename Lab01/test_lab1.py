import  torchvision,torch
import  torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 导入mnist数据集,该数据集是一个手写数字的图像数据集
train = torchvision.datasets.MNIST(root='./mnist/',train=True, transform= transforms.ToTensor(),download=True)
# 使用DataLoader返回一个可迭代的对象，使用for循环便能得到每个批量的数据，批量大小由batch_size指定
dataloader = DataLoader(train, batch_size=50, shuffle=True)

# 查看每次循环返回的数据大小
for step, (x, y) in enumerate(dataloader):
    b_x = x.shape
    b_y = y.shape
    print('Step: ', step, '| train_data的维度' ,b_x,'| train_target的维度',b_y)

# 通过下面函数可以将图像数据可视化
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(titles.numpy()[i])
    return axes

X, y = next(iter(DataLoader(train, batch_size=18,shuffle=True)))
axes = show_images(X.reshape(18, 28, 28), 2, 9, titles=y)
plt.show()

# 试着对每个批量的数据进行中心化，可以考虑使用一个较小的批次来计算均值和标准差，以节省时间和内存。
# 提示：
# transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean, std)
# ])