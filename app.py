from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from medical_image_model.pth import YourModel  # Import your model class

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = YourModel()  # Replace with your model class
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match model input size
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict using the model
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
