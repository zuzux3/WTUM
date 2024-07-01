import torch
import torch.nn as nn
from preprocessing import imgPreprocess
from torchvision import models

def classify(img):
    tensor = imgPreprocess(img)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vgg16 = models.vgg16(weights=None)
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_ftrs, 2)
    checkpointVGG = torch.load('VGG16_model.pth', map_location=device)
    vgg16.load_state_dict(checkpointVGG['model_state_dict'])
    vgg16.to(device)
    vgg16.eval()
    
    with torch.no_grad():
        tensor = tensor.to(device)
        outputVGG = vgg16(tensor)
    _, predictedVGG = torch.max(outputVGG, 1)
        
    
    resnet50 = models.resnet50(weights=None)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 2)
    checkpointResnet = torch.load('ResNet50_model.pth', map_location=device)
    resnet50.load_state_dict(checkpointResnet['model_state_dict'])
    resnet50.to(device)
    resnet50.eval()
    
    with torch.no_grad():
        tensor = tensor.to(device)
        outputResnet = resnet50(tensor)
    _, predictedResnet = torch.max(outputResnet, 1)
    
    predictionVGG = 'Pizza' if predictedVGG.item() == 1 else 'Not Pizza'
    predictionResnet = 'Pizza' if predictedResnet.item() == 1 else 'Not Pizza'
    
    return predictionVGG, predictionResnet
    