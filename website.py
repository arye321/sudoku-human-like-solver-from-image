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
from sudoku.solver_util import SolverUtil


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

def format_sudoku_board(sudoku_string):
    """Convert sudoku string to 9x9 board format"""
    if len(sudoku_string) != 81:
        return None
    
    board = []
    for i in range(0, 81, 9):
        row = [int(digit) if digit != '0' else 0 for digit in sudoku_string[i:i+9]]
        board.append(row)
    return board

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sudoku Board Extractor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
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
            position: relative;
        }
        .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        .submit-btn:disabled {
            background: #a0aec0;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            margin-left: 10px;
        }
        .spinner {
            border: 2px solid #f3f3f3;
            border-top: 2px solid #48bb78;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
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
        .sudoku-board {
            display: inline-block;
            margin: 20px auto;
            border: 3px solid #2d3748;
            background: #2d3748;
            border-radius: 8px;
        }
        .sudoku-row {
            display: flex;
        }
        .sudoku-cell {
            width: 40px;
            height: 40px;
            border: 1px solid #4a5568;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            background: white;
            color: #2d3748;
            transition: all 0.3s ease;
        }
        .sudoku-cell.empty {
            background: #f7fafc;
            color: #a0aec0;
        }
        .sudoku-cell.thick-right {
            border-right: 2px solid #2d3748;
        }
        .sudoku-cell.thick-bottom {
            border-bottom: 2px solid #2d3748;
        }
        .sudoku-cell.hint {
            background: #ffd700 !important;
            animation: pulse-hint 1s ease-in-out;
        }
        .sudoku-cell.solved-hint {
            background: #90EE90 !important;
            animation: pulse-solved 1s ease-in-out;
        }
        @keyframes pulse-hint {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        @keyframes pulse-solved {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .hint-controls {
            margin: 20px 0;
            text-align: center;
        }
        .hint-btn {
            background: linear-gradient(135deg, #ed8936, #dd6b20);
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s;
            margin: 0 10px;
        }
        .hint-btn:hover {
            transform: translateY(-2px);
        }
        .hint-btn:disabled {
            background: #a0aec0;
            cursor: not-allowed;
            transform: none;
        }
        .sudoku-string {
            margin: 15px 0;
            padding: 10px;
            background: #edf2f7;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9em;
            word-break: break-all;
            color: #4a5568;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧩 Sudoku Board Extractor</h1>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="error">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form method="POST" enctype="multipart/form-data" id="upload-form">
            <div class="upload-section">
                <h3>Select a Sudoku image to extract</h3>
                <p>Maximum file size: 5MB<br>
                Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP</p>
                
                <label for="file" class="file-label">
                    📁 Choose Sudoku Image
                </label>
                <input type="file" id="file" name="file" class="file-input" 
                       accept="image/*" onchange="showFileInfo(this)">
                
                <div id="file-info" class="file-info" style="display: none;"></div>
                
                <br>
                <button type="submit" id="submit-btn" class="submit-btn" disabled>
                    <span id="btn-text">🚀 Extract Sudoku Board</span>
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        Processing...
                    </div>
                </button>
            </div>
        </form>
        
        {% if sudoku_board %}
        <div class="result">
            <h3>✅ Sudoku Board Extracted Successfully!</h3>
            
            <div class="sudoku-board">
                {% for row_idx in range(9) %}
                <div class="sudoku-row">
                    {% for col_idx in range(9) %}
                    <div class="sudoku-cell 
                        {% if sudoku_board[row_idx][col_idx] == 0 %}empty{% endif %}
                        {% if col_idx in [2, 5] %}thick-right{% endif %}
                        {% if row_idx in [2, 5] %}thick-bottom{% endif %}"
                        id="cell-{{ row_idx }}-{{ col_idx }}">
                        {% if sudoku_board[row_idx][col_idx] != 0 %}
                            {{ sudoku_board[row_idx][col_idx] }}
                        {% else %}
                            ·
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% endfor %}
            </div>
            
            {% if solving_steps %}
            <div class="hint-controls">
                <button id="hint-btn" class="hint-btn" onclick="toggleHint()">💡 Show Hint</button>
                <button id="reset-btn" class="hint-btn" onclick="resetBoard()" style="background: linear-gradient(135deg, #e53e3e, #c53030);">🔄 Reset Board</button>
            </div>
            {% endif %}
            
            {% if sudoku_string %}
            <div class="sudoku-string">
                <strong>Sudoku String:</strong><br>{{ sudoku_string }}
            </div>
            {% endif %}
            
            {% if image_info %}
            <div style="margin: 15px 0; padding: 10px; background: #e6fffa; border-radius: 5px; font-size: 0.9em;">
                {{ image_info }}
            </div>
            {% endif %}
            <p><small>Processed at: {{ upload_time }}</small></p>
        </div>
        {% endif %}
    </div>

    <script>
        // Store solving steps and current hint index
        {% if solving_steps %}
        const solvingSteps = {{ solving_steps|tojson }};
        console.log('Solving steps:', solvingSteps);
        {% else %}
        const solvingSteps = [];
        {% endif %}
        
        let currentHintIndex = 0;
        let showingHint = false;
        
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
        
        // Handle form submission with loading indicator
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            const submitBtn = document.getElementById('submit-btn');
            const btnText = document.getElementById('btn-text');
            const loading = document.getElementById('loading');
            
            // Show loading state
            btnText.style.display = 'none';
            loading.style.display = 'inline-block';
            submitBtn.disabled = true;
        });
        
        function toggleHint() {
            console.log('Toggle hint called, currentHintIndex:', currentHintIndex, 'total steps:', solvingSteps.length);
            
            if (currentHintIndex >= solvingSteps.length) {
                document.getElementById('hint-btn').textContent = '🎉 All Hints Complete!';
                document.getElementById('hint-btn').disabled = true;
                return;
            }
            
            const hintBtn = document.getElementById('hint-btn');
            const currentStep = solvingSteps[currentHintIndex];
            console.log('Current step:', currentStep);
            
            // Handle different data formats
            let row, col, value;
            if (Array.isArray(currentStep)) {
                row = currentStep[0];
                col = currentStep[1];
                value = currentStep[2];
            } else if (typeof currentStep === 'object') {
                // If it's an object, try different property names
                row = currentStep.row || currentStep.r || currentStep[0];
                col = currentStep.col || currentStep.c || currentStep[1];
                value = currentStep.value || currentStep.val || currentStep.v || currentStep[2];
            } else {
                console.error('Unexpected step format:', currentStep);
                return;
            }
            
            console.log('Parsed values - row:', row, 'col:', col, 'value:', value);
            
            const cellId = `cell-${row}-${col}`;
            const cell = document.getElementById(cellId);
            
            if (!cell) {
                console.error('Cell not found:', cellId);
                return;
            }
            
            if (!showingHint) {
                // Show hint (highlight the cell)
                cell.classList.add('hint');
                cell.classList.remove('solved-hint');
                hintBtn.textContent = `✨ Solve Hint `;
                showingHint = true;
                console.log('Showing hint for cell:', cellId);
            } else {
                // Solve hint (show the value)
                cell.classList.remove('hint');
                cell.classList.add('solved-hint');
                cell.textContent = value;
                cell.classList.remove('empty');
                
                currentHintIndex++;
                showingHint = false;
                
                if (currentHintIndex < solvingSteps.length) {
                    hintBtn.textContent = '💡 Show Hint';
                } else {
                    hintBtn.textContent = '🎉 All Hints Complete!';
                    hintBtn.disabled = true;
                }
                
                console.log('Solved hint for cell:', cellId, 'with value:', value);
                
                // Remove the solved-hint class after animation
                setTimeout(() => {
                    cell.classList.remove('solved-hint');
                }, 1000);
            }
        }
        
        function resetBoard() {
            // Reset all cells to original state
            const originalBoard = {{ sudoku_board|tojson }};
            
            for (let row = 0; row < 9; row++) {
                for (let col = 0; col < 9; col++) {
                    const cellId = `cell-${row}-${col}`;
                    const cell = document.getElementById(cellId);
                    
                    // Reset classes
                    cell.classList.remove('hint', 'solved-hint');
                    
                    // Reset content
                    if (originalBoard[row][col] === 0) {
                        cell.textContent = '·';
                        cell.classList.add('empty');
                    } else {
                        cell.textContent = originalBoard[row][col];
                        cell.classList.remove('empty');
                    }
                }
            }
            
            // Reset hint state
            currentHintIndex = 0;
            showingHint = false;
            document.getElementById('hint-btn').textContent = '💡 Show Hint';
            document.getElementById('hint-btn').disabled = false;
        }
    </script>
</body>
</html>
'''

def process_img(img):
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
                return "-1", []
    
                

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
    
    solver ="Strategic"
    solving_steps = SolverUtil.solve_puzzle(
            sudoku_string,
            verbose=False,
            description='xd',
            solver_type=solver,
        )
    return (sudoku_string, solving_steps['inserted_values'])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    sudoku_board = None
    sudoku_string = None
    solving_steps = None
    upload_time = None
    image_info = None
    
    if request.method == 'POST':
        # Check if file was submitted
        if 'file' not in request.files:
            flash('No file selected!')
            return redirect(url_for('upload_file'))
        
        file = request.files['file']
        
        # Check if file was actually selected
        if file.filename == '':
            flash('No file selected!')
            return redirect(url_for('upload_file'))
        
        if file and allowed_file(file.filename):
            # Validate that it's actually an image
            file_ext = validate_image(file.stream)
            if not file_ext:
                flash('Invalid image file! Please upload a valid image.')
                return redirect(url_for('upload_file'))

            
            try:
                # Load image directly into OpenCV without saving to disk
                file_bytes = np.frombuffer(file.read(), np.uint8)
                cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if cv_image is None:
                    flash('Could not process image file!')
                    return redirect(url_for('upload_file'))

                
                # Process the image to extract sudoku
                result = process_img(cv_image)
                if result[0] != "-1":
                    sudoku_string, solving_steps = result
                    sudoku_board = format_sudoku_board(sudoku_string)
                
                if sudoku_board is None:
                    flash('Could not extract valid sudoku board from image!')
                    return redirect(url_for('upload_file'))

                
                # Get image dimensions
                height, width, channels = cv_image.shape
                image_info = f"Image processed successfully! Dimensions: {width}x{height}, Channels: {channels}"
                
                # Generate timestamp
                now = datetime.now()
                upload_time = now.strftime("%B %d, %Y at %I:%M:%S %p")
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(url_for('upload_file'))
            
        else:
            flash('Invalid file type! Please upload an image file (PNG, JPG, JPEG, GIF, BMP, WEBP).')
            return redirect(url_for('upload_file'))
    
    return render_template_string(HTML_TEMPLATE, 
                                sudoku_board=sudoku_board,
                                sudoku_string=sudoku_string,
                                solving_steps=solving_steps,
                                upload_time=upload_time,
                                image_info=image_info)

@app.errorhandler(413)
def too_large(e):
    flash('File too large! Maximum size is 5MB.')
    return redirect(url_for('upload_file'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)