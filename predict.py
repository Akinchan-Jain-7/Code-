import torch
from torchvision import transforms, models
from PIL import Image
import io

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 8)
model.load_state_dict(torch.load('./model/fingerprint_blood_group_resnet.pth', map_location='cpu'))
model.eval()

# Blood group labels
CLASS_NAMES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Transform for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_blood_group(image_file):
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]
