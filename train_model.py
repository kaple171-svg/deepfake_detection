import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from skimage.measure import shannon_entropy

def extract_features(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB (from BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    features = []
    
    # 1. Enhanced color statistics (mean, std, skewness)
    for channel in range(3):
        channel_data = img_rgb[:,:,channel].flatten()
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.mean((channel_data - np.mean(channel_data))**3) / (np.std(channel_data)**3)  # skewness
        ])
        
    # Add color correlation features
    correlations = np.corrcoef(img_rgb.reshape(-1, 3).T)
    features.extend(correlations[np.triu_indices(3, k=1)])  # Upper triangle of correlation matrix
    
    # 2. Noise analysis and texture features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Local Binary Pattern for texture analysis
    from skimage.feature import local_binary_pattern
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
    features.extend(lbp_hist)
    
    # Multi-scale gradient analysis
    for scale in [1, 2, 4]:
        scaled = cv2.resize(gray, None, fx=1/scale, fy=1/scale)
        gx = cv2.Sobel(scaled, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(scaled, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features.extend([
            np.mean(grad_mag),
            np.std(grad_mag),
            np.percentile(grad_mag, 90)
        ])
    
    # 3. Noise estimation using multiple filter scales
    for ksize in [3, 5, 7]:
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        noise = gray.astype(np.float32) - blurred
        features.extend([
            np.std(noise),
            np.mean(np.abs(noise)),
            np.percentile(np.abs(noise), 95)
        ])
    
    # 4. Advanced edge analysis
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze contour properties
    if len(contours) > 0:
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        contour_lengths = [cv2.arcLength(cnt, True) for cnt in contours]
        features.extend([
            len(contours),  # Number of edges
            np.mean(contour_areas),  # Average edge region size
            np.std(contour_areas),   # Variation in edge regions
            np.mean(contour_lengths), # Average edge length
            np.std(contour_lengths)   # Variation in edge lengths
        ])
    else:
        features.extend([0, 0, 0, 0, 0])
    
    # 5. Image entropy
    entropy = shannon_entropy(gray)
    features.append(entropy)
    
    return np.array(features)

def train_model(real_dir, fake_dir):
    features = []
    labels = []
    
    # Process real images
    for img_name in os.listdir(real_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(real_dir, img_name)
                feat = extract_features(img_path)
                features.append(feat)
                labels.append(1)  # 1 for real
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
    
    # Process fake images
    for img_name in os.listdir(fake_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(fake_dir, img_name)
                feat = extract_features(img_path)
                features.append(feat)
                labels.append(0)  # 0 for fake
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train SVM with optimized parameters
    from sklearn.model_selection import cross_val_score
    
    # Define multiple parameter combinations
    param_combinations = [
        {'kernel': 'rbf', 'C': 100.0, 'gamma': 'scale', 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'auto', 'class_weight': 'balanced'},
        {'kernel': 'poly', 'C': 10.0, 'degree': 3, 'class_weight': 'balanced'}
    ]
    
    best_score = 0
    best_params = None
    
    # Find best parameters
    for params in param_combinations:
        clf = SVC(probability=True, random_state=42, **params)
        scores = cross_val_score(clf, features_scaled, labels, cv=2, scoring='accuracy')
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    # Train final model with best parameters
    svm = SVC(probability=True, random_state=42, **best_params)
    svm.fit(features_scaled, labels)
    
    # Evaluate training results
    train_pred = svm.predict(features_scaled)
    train_prob = svm.predict_proba(features_scaled)
    accuracy = (train_pred == labels).mean()
    print(f"\nTraining accuracy: {accuracy * 100:.2f}%")
    
    # Print detailed predictions
    print("\nDetailed training results:")
    for i, (true_label, pred, probs) in enumerate(zip(labels, train_pred, train_prob)):
        img_type = "real" if true_label == 1 else "fake"
        pred_type = "real" if pred == 1 else "fake"
        # Map probabilities according to svm.classes_
        try:
            classes = svm.classes_.tolist()
            real_idx = classes.index(1) if 1 in classes else None
            fake_idx = classes.index(0) if 0 in classes else None
            real_prob = probs[real_idx] * 100 if real_idx is not None else 0.0
            fake_prob = probs[fake_idx] * 100 if fake_idx is not None else 0.0
        except Exception:
            real_prob = probs[1] * 100 if len(probs) > 1 else 0.0
            fake_prob = probs[0] * 100 if len(probs) > 0 else 0.0
        print(f"Image {i+1} (True: {img_type}):")
        print(f"  Predicted: {pred_type}")
        print(f"  Real probability: {real_prob:.2f}%")
        print(f"  Fake probability: {fake_prob:.2f}%")
    
    # Print predictions for each image
    print("\nPredictions for training images:")
    for i, (pred, true_label) in enumerate(zip(train_pred, labels)):
        img_type = "real" if true_label == 1 else "fake"
        pred_type = "real" if pred == 1 else "fake"
        print(f"Image {i+1} (True: {img_type}): Predicted as {pred_type}")
    
    # Save model and scaler
    joblib.dump(svm, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return svm, scaler

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--real', required=True, help='Path to directory containing real images')
    parser.add_argument('--fake', required=True, help='Path to directory containing fake images')
    
    args = parser.parse_args()
    
    # Verify directories exist
    if not os.path.exists(args.real):
        print(f"Error: Real images directory '{args.real}' does not exist!")
        exit(1)
    if not os.path.exists(args.fake):
        print(f"Error: Fake images directory '{args.fake}' does not exist!")
        exit(1)
        
    print(f"Using real images from: {args.real}")
    print(f"Using fake images from: {args.fake}")
    print("\nTraining model...")
    
    try:
        model, scaler = train_model(args.real, args.fake)
        print("\nModel and scaler saved successfully!")
        print("You can now run the Flask application (app.py)")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        exit(1)