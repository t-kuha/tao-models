import json
import torch
import torchvision

with open('classmap.json', 'r') as f:
    data = json.load(f)
labels = [k for k, v in data.items()]

# transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ConvertImageDtype(torch.float),
    torchvision.transforms.Normalize(mean=[103.939 / 255, 116.779 / 255, 123.68 / 255], std=[1, 1, 1])
])
model = torch.load('resnet_18.pth', weights_only=False)

img_path = 'ffae2f74dd51806b.jpg'
img = torchvision.io.read_image(img_path)

pred = model(transform(img) * 255)

top5 = torch.topk(pred, 5)
for i, t in enumerate( top5.indices[0]):
    print(f'{i}: {labels[t]}')
