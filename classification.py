import torch
import torch.nn as nn
from preprocessing import imgPreprocess
from torchvision import models

def classify(img):
    tensor = imgPreprocess(img)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resnet18 = models.resnet18(weights=None)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)
    resnet18Checkpoint = models.resnet18(torch.load('resnet18.pth', map_location=device))
    resnet18.to(device)
    resnet18.eval()
    
    resnet50 = models.resnet50(weights=None)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 2)
    resnet50Checkpoint = models.resnet50(torch.load('resnet50.pth', map_location=device))
    resnet50.to(device)
    resnet50.eval()
    
    with torch.no_grad():
        tensor = tensor.to(device)
        outputRes18 = resnet18(tensor)
        
    _, predictedRes18 = torch.max(outputRes18, 1)
    
    with torch.no_grad():
        tensor = tensor.to(device)
        outputRes50 = resnet50(tensor)
        
    _, predictedRes50 = torch.max(outputRed50, 1)
    
    predictedRes18 = 'Not Pizza' if predictedRes18.item() == 0 else 'Pizza'
    predictedRes50 = 'Not Pizza' if predictedRes50.item() == 0 else 'Pizza'
    
    return predictedRes18, predictedRes50
     