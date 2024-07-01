import torch
import torch.nn as nn
from preprocessing import imgPreprocess
from torchvision import models

def loadModelWeights(model, checkpointPath, device):
    checkpoint = torch.load(checkpointPath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['stateDict'])
    model.eval()
    return model

def classify(img):
    tensor = imgPreprocess(img)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resnet18 = models.resnet18(weights=None)
    resnet18 = loadModelWeights(resnet18, 'resnet18.pth', device)
    
    resnet50 = models.resnet18(weights=None)
    resnet50 = loadModelWeights(resnet50, 'resnet50.pth', device)
    
    
    model50 = models.resnet50(weights=None)
    with torch.no_grad():
        tensor = tensor.to(device)
        outputRes18 = resnet18(tensor)
        
    _, predictedRes18 = torch.max(outputRes18, 1)
    
    with torch.no_grad():
        tensor = tensor.to(device)
        outputRes50 = resnet50(tensor)
        
    _, predictedRes50 = torch.max(outputRes50, 1)
    
    predictedRes18 = 'Not Pizza' if predictedRes18.item() == 0 else 'Pizza'
    predictedRes50 = 'Not Pizza' if predictedRes50.item() == 0 else 'Pizza'
    
    return predictedRes18, predictedRes50
     