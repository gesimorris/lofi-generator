"""
Test script to verify backend is working
"""
import sys
sys.path.append('backend')

import numpy as np
from pathlib import Path

print("Testing backend setup...\n")

# Test 1: Check if model files exist
print("1. Checking model files...")
model_path = Path("backend/models/lofi_model.npy")
scaler_x_path = Path("backend/models/scaler_x.npy")
scaler_y_path = Path("backend/models/scaler_y.npy")

if model_path.exists():
    print(f"   ✅ Model found: {model_path}")
else:
    print(f"   ❌ Model NOT found: {model_path}")

if scaler_x_path.exists():
    print(f"   ✅ Scaler X found: {scaler_x_path}")
else:
    print(f"   ❌ Scaler X NOT found: {scaler_x_path}")

if scaler_y_path.exists():
    print(f"   ✅ Scaler Y found: {scaler_y_path}")
else:
    print(f"   ❌ Scaler Y NOT found: {scaler_y_path}")

# Test 2: Try loading the model
print("\n2. Testing model loading...")
try:
    from backend.improved_model import ImprovedNeuralNetwork
    model = ImprovedNeuralNetwork(input_size=6, hidden_sizes=[64, 128, 128, 64], output_size=20)
    model.load_model(str(model_path))
    print("   ✅ Model loaded successfully!")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")

# Test 3: Try loading scalers
print("\n3. Testing scaler loading...")
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    scaler_x_data = np.load(scaler_x_path, allow_pickle=True).item()
    scaler_x = StandardScaler()
    scaler_x.mean_ = scaler_x_data['mean']
    scaler_x.scale_ = scaler_x_data['scale']
    scaler_x.var_ = scaler_x_data['var']
    scaler_x.n_features_in_ = len(scaler_x.mean_)
    print(f"   ✅ Scaler X loaded! Features: {scaler_x.n_features_in_}")
    
    scaler_y_data = np.load(scaler_y_path, allow_pickle=True).item()
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.min_ = scaler_y_data['min']
    scaler_y.scale_ = scaler_y_data['scale']
    scaler_y.data_min_ = scaler_y_data['data_min']
    scaler_y.data_max_ = scaler_y_data['data_max']
    scaler_y.n_features_in_ = len(scaler_y.min_)
    print(f"   ✅ Scaler Y loaded! Features: {scaler_y.n_features_in_}")
except Exception as e:
    print(f"   ❌ Error loading scalers: {e}")

# Test 4: Test a prediction
print("\n4. Testing prediction...")
try:
    # Create dummy image features
    dummy_features = np.array([120.0, 50.0, 130.0, 125.0, 115.0, 0.05])
    dummy_features_2d = dummy_features.reshape(1, -1)
    dummy_scaled = scaler_x.transform(dummy_features_2d)
    
    prediction = model.predict(dummy_scaled)
    print(f"   ✅ Prediction successful! Shape: {prediction.shape}")
    print(f"   Sample output: {prediction[0][:5]}...")
except Exception as e:
    print(f"   ❌ Error during prediction: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Backend test complete!")
print("="*50)
