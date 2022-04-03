from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms



img_path = "D:/PythonML/PytorchDemo/hymenoptera_data/train/ants/6743948_2b8c096dda.jpg"
img = Image.open(img_path)

# ToTensor
tensor = transforms.ToTensor()
tensor_img = tensor(img)
print(tensor_img)

# Normalize

# before normalization
print(tensor_img[0][0][0])
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
norm_tensor = normalize(tensor_img)
# after normalization
print(norm_tensor[0][0][0])




