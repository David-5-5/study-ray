import os
import torch
import torchvision.models as models

import os

model_dir = os.path.expanduser("~/.models")
os.makedirs(model_dir, exist_ok=True)

weight_path = os.path.join(model_dir, "model_weights.pth")

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), weight_path)

model = models.vgg16()
model.load_state_dict(torch.load(weight_path, weights_only=True))

model.eval()

model_path = os.path.join(model_dir, "model.pth")

torch.save(model, model_path)

model = torch.load(model_path, weights_only=False)
