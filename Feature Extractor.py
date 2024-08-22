import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba_array
from torchinfo import summary
from thop import profile

import torch
import torchvision.transforms as transforms
from PIL import Image

def hook_fn(module, input, output):
    global intermediate_features
    intermediate_features.append(output)

def extract_features(model, img, layer_index):
    global intermediate_features
    intermediate_features = []
    hook = model.model.model[layer_index].register_forward_hook(hook_fn)
    # macs, params = profile(model, inputs=(img))
    with torch.no_grad():
        model(img)
    hook.remove()
    return intermediate_features[0]

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img

intermediate_features = []
layer_index = 19

weights_path = Path("./POT-YOLO.pt")
model = YOLO(weights_path)
model.to('cuda')
img = Path(r'./pothole.png')

img = preprocess_image(img)
features = extract_features(model, img, layer_index)


def extract_and_plot_features(img_path, layer_index, channel_index=5):
    
    img = preprocess_image(img_path)
    features = extract_features(model, img, layer_index)

    print(f"Features shape for {img_path.name}: {features.shape}")

    plt.figure(figsize=(10, 5))
    sns.heatmap(features[0][channel_index].cpu().numpy(), cmap='viridis', annot=False)
    plt.title(f'Features for {img_path.name} - Layer {layer_index} - Channel {channel_index}')
    plt.show()

image = Path(r"./Airplane.png") 
extract_and_plot_features(image, layer_index)

summary(model)


# import torch

# x = torch.cuda.is_available()
# print(x)

# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')

# model.predict(source="Airplane.png", device='cuda', show=True, save=True, conf=0.45)