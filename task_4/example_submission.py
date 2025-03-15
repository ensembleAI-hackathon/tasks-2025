import base64
import io
import json
import numpy as np
import pickle
import requests
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from typing import Tuple
from torchvision import models


TOKEN = ...                         # Your token here
SUBMIT_URL = "149.156.182.9:6060/task-4/submit"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "model.pth"



def submitting_example():

    # Create a dummy model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 4)
    torch.save(model.state_dict(), CHECKPOINT_PATH)

    response = requests.post(SUBMIT_URL, headers={"token": TOKEN}, files={"model_state_dict": open(CHECKPOINT_PATH, "rb")})
    print(response.status_code, response.text)



if __name__ == '__main__':
    submitting_example()

