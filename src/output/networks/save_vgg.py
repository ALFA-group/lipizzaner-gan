import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16

model = vgg16()
# Change the fc2 from (4096, 4096) to (4096, 64). Remove the last fc layer.
for out_dim in [10, 32, 64]:
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-4], nn.Linear(4096, out_dim))
    model._initialize_weights()
    torch.save(model.state_dict(), f"random_vgg16_{out_dim}.pth")

