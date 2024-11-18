# app.py (or main.py if that's what you're using)
from flask import Flask, request, jsonify, send_from_directory
from app.utils.cnn_api import CNN

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # Folder for uploaded images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the CNN model
cnn = CNN(architecture='deep-wide',
           input_tensors="app/utils/data/label_tensors.pt",
           model_path='app/utils/data/model_11_12_dw.pth'
        )

@app.route('/api/process_image', methods=['POST'])
def process_image():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    category = cnn.predict_image_tensor(file)
    return jsonify({'category': category})

@app.route('/')
def home():
    return send_from_directory('frontend/static', 'predict_image.html')


 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
