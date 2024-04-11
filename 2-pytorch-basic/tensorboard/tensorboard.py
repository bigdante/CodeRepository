# 这种方式已经不行了
# from torch.utils.tensorboard import SummaryWriter
# 要用这种方式导入SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

image_path = "800px-Meat_eater_ant_qeen_excavating_hole.jpg"
image_array = np.array(Image.open(image_path))
writer.add_image("image", image_array, 2, dataformats='HWC')
writer.close()
# 运行命令，进入网页查看，可以修改端口
# tensorboard --logdir=logs --port=6007
