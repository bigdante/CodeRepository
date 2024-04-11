from PIL import Image
from torchvision import transforms
from tensorboardX import SummaryWriter

image_path = "/2-pytorch_nn_rnn/tensorboard/tensorboard_data/train/ants/0013035.jpg"
imag = Image.open(image_path)
to_tensor = transforms.ToTensor()

image_tensor = to_tensor(pic=imag)

writer = SummaryWriter("logs")
# normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
imag_norm = trans_norm(image_tensor)
writer.add_image("image", image_tensor)
writer.add_image("image_norm", imag_norm)

# resize the image

trans_resize = transforms.Resize((2000, 3000))
image_resize = trans_resize(imag)
image_resize = to_tensor(image_resize)
writer.add_image("image_resize1", image_resize)

# 用compose对图像进行处理
trans_resize2 = transforms.Resize(512)
# compose 为了把多个步骤放在一起，例如这里是将resize和to_tensor按照流程放进去
# 因此前面的trans_resize2输出类型，必须要符合后面的to_tensor输入
trans_compose = transforms.Compose([trans_resize2, to_tensor])
image_resize2 = trans_compose(imag)
writer.add_image("resize2", image_resize2, 1)

writer.close()
