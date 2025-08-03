from flask import Flask, request, render_template_string, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import imghdr
import cv2
import numpy as np

import torch
from digits_classifier.helper_functions_pt import MNISTClassifier, get_mnist_transform

import imutils
from image_processing import get_grid_dimensions, filter_non_square_contours, sort_grid_contours, reduce_noise, transform_grid, get_cells_from_9_main_cells
from csp import csp, create_empty_board, BLANK_STATE
from digits_classifier import sudoku_cells_reduce_noise
from PIL import Image

device = "cpu"
model = MNISTClassifier().to(device)

state_dict = torch.load("digits_classifier/models/pt_cnn/ft_model_epoch12.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(stream):
    """Validate that the uploaded file is actually an image"""
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Upload - DateTime Generator</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #4a5568;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .upload-section {
            border: 3px dashed #cbd5e0;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 30px 0;
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            border-color: #667eea;
            background-color: #f7fafc;
        }
        .file-input {
            display: none;
        }
        .file-label {
            display: inline-block;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1em;
            transition: transform 0.2s;
        }
        .file-label:hover {
            transform: translateY(-2px);
        }
        .submit-btn {
            background: linear-gradient(135deg, #48bb78, #38a169);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 20px;
        }
        .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        .submit-btn:disabled {
            background: #a0aec0;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f0fff4;
            border: 2px solid #48bb78;
            border-radius: 8px;
            text-align: center;
        }
        .error {
            margin-top: 20px;
            padding: 15px;
            background: #fed7d7;
            border: 2px solid #e53e3e;
            border-radius: 8px;
            color: #c53030;
        }
        .file-info {
            margin-top: 15px;
            padding: 10px;
            background: #edf2f7;
            border-radius: 5px;
            font-size: 0.9em;
            color: #4a5568;
        }
        .datetime-display {
            font-size: 1.5em;
            font-weight: bold;
            color: #2d3748;
            margin: 15px 0;
            padding: 15px;
            background: linear-gradient(135deg, #e6fffa, #b2f5ea);
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 Image Upload & DateTime Generator</h1>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="error">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form method="POST" enctype="multipart/form-data">
            <div class="upload-section">
                <h3>Select an image to upload</h3>
                <p>Maximum file size: 5MB<br>
                Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP</p>
                
                <label for="file" class="file-label">
                    📁 Choose Image File
                </label>
                <input type="file" id="file" name="file" class="file-input" 
                       accept="image/*" onchange="showFileInfo(this)">
                
                <div id="file-info" class="file-info" style="display: none;"></div>
                
                <br>
                <button type="submit" id="submit-btn" class="submit-btn" disabled>
                    🚀 Upload & Generate DateTime
                </button>
            </div>
        </form>
        
        {% if datetime_result %}
        <div class="result">
            <h3>✅ Upload Successful!</h3>
            <div class="datetime-display">
                Generated DateTime: {{ datetime_result }}
            </div>
            {% if image_info %}
            <div style="margin: 15px 0; padding: 10px; background: #e6fffa; border-radius: 5px; font-size: 0.9em;">
                {{ image_info }}
            </div>
            {% endif %}
            <p><small>Timestamp generated at: {{ upload_time }}</small></p>
        </div>
        {% endif %}
    </div>

    <script>
        function showFileInfo(input) {
            const fileInfo = document.getElementById('file-info');
            const submitBtn = document.getElementById('submit-btn');
            
            if (input.files && input.files[0]) {
                const file = input.files[0];
                const fileSize = (file.size / (1024 * 1024)).toFixed(2);
                
                fileInfo.innerHTML = `
                    <strong>Selected file:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${fileSize} MB<br>
                    <strong>Type:</strong> ${file.type}
                `;
                fileInfo.style.display = 'block';
                submitBtn.disabled = false;
                
                if (file.size > 5 * 1024 * 1024) {
                    fileInfo.innerHTML += '<br><span style="color: red;">⚠️ File too large! Maximum 5MB allowed.</span>';
                    submitBtn.disabled = true;
                }
            } else {
                fileInfo.style.display = 'none';
                submitBtn.disabled = true;
            }
        }
    </script>
</body>
</html>
'''
def prosses_img(img):
    image = img
    if image.shape[1] > 700:
        image = imutils.resize(image, width=700)
    grid_coordinates = get_grid_dimensions(image)




    grid = transform_grid(image, grid_coordinates)

    # Image preprocessing, reduce noise such as numbers/dots, cover all numbers
    thresh = reduce_noise(grid)

    # Contour detection again, this time we are extracting the grid
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out non square contours
    cnts = filter_non_square_contours(cnts)

    # Convert contours into data to work with
    # Check how many valid cnts are found
    if 9 <= (cnts_len := len(cnts)) <= 90:
        # Salvageable
        if cnts_len == 81:
            # All cells extracted, perfect
            pass
        elif cnts_len == 9:
            # Split main cells to 81 cells
            cnts = get_cells_from_9_main_cells(cnts)
        else:
            new_cnts = []

            # In between, not sure if this is a valid grid
            # Sort hierarchy, toss small contours to find main cells
            # Only accept contours with hierarchy 0 (main contours)
            # Format of hierarchy: [next, previous, child, parent]
            for cnt, hie in zip(cnts, hierarchy[0]):
                # Check if parent is -1 (Does not exist)
                if hie[3] == -1:
                    new_cnts.append(cnt)

            if len(new_cnts) == 9:
                # Got all main cells
                cnts = get_cells_from_9_main_cells(new_cnts)
            else:
                # Unable to identify main cells
                raise Exception(f"File: {file_name}, Unable to extract grid cells properly")
    
                

        # Finally
        # Update contour len, in case any contour filtering/adjustment was made
        cnts_len = len(cnts)

        # Success detection of grid & cells
        # Sort grid into nested list format same as sudoku
        grid_contours = sort_grid_contours(cnts)

        # Create a blank Sudoku board
        board = create_empty_board()

        # Run digit classifier
        for row_index, row in enumerate(grid_contours):
            for box_index, box in enumerate(row):

                # Extract cell ROI from contour
                x, y, width, height = cv2.boundingRect(box)
                roi = grid[y:y + height, x:x + width]

                # Convert to greyscale
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Image thresholding & invert image
                digit_inv = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 27, 11)

                # Remove surrounding noise
                digit = sudoku_cells_reduce_noise(digit_inv)

                # Digit present
                if digit is not None:
                    # Reshape to fit model input
                    # digit = digit.reshape((1, 28, 28, 1))
                    # Make prediction
                    # board[row_index][box_index] = np.argmax(model.predict(digit), axis=-1)[0] + 1

                    # Make prediction
                    digit = Image.fromarray(digit)
                    # Reshape to fit model input, [1,28,28]
                    digit_tensor = get_mnist_transform()(digit)
                    # Add batch dim, send to device
                    digit_tensor = digit_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(digit_tensor)
                        board[row_index][box_index] = torch.argmax(logits, dim=1).item() + 1
    sudoku_string = ''                    
    for row in board:
        for digit in row:
            sudoku_string += str(digit)
    return sudoku_string

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    datetime_result = None
    upload_time = None
    
    if request.method == 'POST':
        # Check if file was submitted
        if 'file' not in request.files:
            flash('No file selected!')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file was actually selected
        if file.filename == '':
            flash('No file selected!')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Validate that it's actually an image
            file_ext = validate_image(file.stream)
            if not file_ext:
                flash('Invalid image file! Please upload a valid image.')
                return redirect(request.url)
            
            try:
                # Load image directly into OpenCV without saving to disk
                file_bytes = np.frombuffer(file.read(), np.uint8)
                cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if cv_image is None:
                    flash('Could not process image file!')
                    return redirect(request.url)
                sudoku_string = prosses_img(cv_image)
                # Example: Get image dimensions using OpenCV
                height, width, channels = cv_image.shape
                image_info = f"Image loaded successfully! Dimensions: {width}x{height}, Channels: {channels}, string={sudoku_string}"
                
                # You can now perform any OpenCV operations on cv_image
                # Example operations (uncomment to use):
                # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                # blurred = cv2.GaussianBlur(cv_image, (15, 15), 0)
                # edges = cv2.Canny(cv_image, 100, 200)
                
                # Generate datetime string
                now = datetime.now()
                datetime_result = now.strftime("%Y-%m-%d %H:%M:%S")
                upload_time = now.strftime("%B %d, %Y at %I:%M:%S %p")
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
            
        else:
            flash('Invalid file type! Please upload an image file (PNG, JPG, JPEG, GIF, BMP, WEBP).')
            return redirect(request.url)
    
    return render_template_string(HTML_TEMPLATE, 
                                datetime_result=datetime_result, 
                                upload_time=upload_time,
                                image_info=image_info if 'image_info' in locals() else None)

@app.errorhandler(413)
def too_large(e):
    flash('File too large! Maximum size is 5MB.')
    return redirect(url_for('upload_file'))


if __name__ == '__main__':
    # Create uploads directory if you want to save files
    # os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)