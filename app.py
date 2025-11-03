import os
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import joblib
import base64
from werkzeug.utils import secure_filename
from train_model import extract_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model and scaler
MODEL_PATH = 'model.joblib'
SCALER_PATH = 'scaler.joblib'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        model = None
        scaler = None
else:
    print("Model files not found. Please train the model first.")
    model = None
    scaler = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                if model is None or scaler is None:
                    return render_template('index.html', 
                                         error="Model not loaded. Please train the model first by running train_model.py")

                # Extract features
                features = extract_features(filepath)
                
                # Scale features
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Get prediction
                prediction = model.predict(features_scaled)[0]

                # Prefer a probability derived from the decision function to ensure
                # consistency with the predicted label (predict uses the sign of
                # decision_function). Use a sigmoid of the decision value for a
                # calibrated-like probability. Fall back to predict_proba when
                # decision_function is unavailable.
                real_prob = None
                fake_prob = None
                try:
                    # decision_function returns positive values for the class
                    # considered the positive class. For binary SVC this is a
                    # single float per sample.
                    dec = model.decision_function(features_scaled)
                    # dec may be array-like; get scalar
                    if hasattr(dec, '__len__'):
                        dec_val = float(dec[0])
                    else:
                        dec_val = float(dec)
                    # sigmoid to map to (0,1)
                    import math
                    sigmoid = 1.0 / (1.0 + math.exp(-dec_val))
                    # determine which index corresponds to label 1 (real)
                    classes = getattr(model, 'classes_', None)
                    if classes is not None:
                        classes = list(classes)
                        # If label 1 is the "positive" class used by decision,
                        # sigmoid(dec) approximates P(label==1).
                        if 1 in classes:
                            real_prob = float(sigmoid * 100)
                        else:
                            # if labels are not 0/1, assume sigmoid gives prob
                            # for classes[1]
                            real_prob = float(sigmoid * 100)
                    else:
                        real_prob = float(sigmoid * 100)
                    fake_prob = 100.0 - real_prob
                except Exception:
                    # fallback to predict_proba if decision_function not present
                    try:
                        probabilities = model.predict_proba(features_scaled)[0]
                        classes = model.classes_.tolist()
                        real_idx = classes.index(1) if 1 in classes else None
                        fake_idx = classes.index(0) if 0 in classes else None
                        real_prob = float(probabilities[real_idx] * 100) if real_idx is not None else float(probabilities[1] * 100)
                        fake_prob = float(probabilities[fake_idx] * 100) if fake_idx is not None else float(probabilities[0] * 100)
                    except Exception:
                        # final fallback
                        real_prob = 0.0
                        fake_prob = 0.0
                
                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except Exception:
                    pass

                result = {
                    'prediction': 'Real' if prediction == 1 else 'Fake',
                    'real_probability': real_prob,
                    'fake_probability': fake_prob
                }

                # attempt to use request.file.read() for image_data (fallback to empty)
                try:
                    img_bytes = file.read()
                    # if file.read() returns bytes, encode; else set empty
                    if isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
                        image_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    else:
                        image_b64 = ''
                except Exception:
                    image_b64 = ''

                return render_template('result.html', 
                                     result=result,
                                     image_data=image_b64)
            
            except Exception as e:
                return jsonify({'error': str(e)})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)