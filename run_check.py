import joblib
from train_model import extract_features

MODEL='model.joblib'
SCALER='scaler.joblib'
IMG=r'C:\Users\morej\OneDrive\Desktop\jikky\dataset\real_images\real1.jpg'

m=joblib.load(MODEL)
s=joblib.load(SCALER)
f=extract_features(IMG)
fs=f.reshape(1,-1)
fs_s=s.transform(fs)
pre=m.predict(fs_s)[0]
probs=m.predict_proba(fs_s)[0]
dec=m.decision_function(fs_s)
classes=m.classes_.tolist()
real_idx=classes.index(1) if 1 in classes else None
fake_idx=classes.index(0) if 0 in classes else None
print('classes:', classes)
print('predicted_label:', pre)
print('real_prob:', probs[real_idx]*100 if real_idx is not None else None)
print('fake_prob:', probs[fake_idx]*100 if fake_idx is not None else None)
print('decision_function:', dec)
