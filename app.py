import streamlit as st
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# Define transformations for inference

with open('styles.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

data_transforms = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load class names
class_names = ['ai_images', 'real_images']

# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Load the pre-trained model
model = SimpleCNN(num_classes=len(class_names))
model.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# Function to predict and display an image
def predict_image(model, image, class_names):
    try:
        image = data_transforms(image).unsqueeze(0)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return class_names[preds.item()]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Streamlit app
st.title("AI Image Classifier")
st.write("Upload an image and the model will predict whether it's AI-generated or a real image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    with st.spinner('Classifying...'):
        prediction = predict_image(model, image, class_names)
        if prediction:
            st.markdown(f'<p class="prediction">Predicted: {prediction}</p>', unsafe_allow_html=True)

