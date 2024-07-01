import numpy as np
from PIL import Image
from torchvision import transforms

def imgPreprocess(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])
    
    tensor = transform(img)
    batch = tensor.unsqueeze(0)
    
    return batch