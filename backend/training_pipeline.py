"""
Complete Training Pipeline
Handles data loading, preprocessing, augmentation, training, and model evaluation
"""

import numpy as np
import cv2
import mido
import os
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import Counter

from improved_model import ImprovedNeuralNetwork
from data_augmentation import run_augmentation


# Constants for MIDI processing
TEMPO_RANGE_CLAMP = (50, 160)
MIDI_PITCH_RANGE_CLAMP = (36, 96)
VELOCITY_RANGE_CLAMP = (30, 100)
P_RANGE_CLAMP = (0, 48)
P_STD_CLAMP = (0, 12)
R_DENSITY_CLAMP = (0.1, 10)
AVG_DUR_CLAMP = (0.1, 4.0)
DUR_STD_CLAMP = (0, 2.0)


def extract_image_features(image_path, canny_low_threshold=50, canny_high_threshold=150):
    """
    Extract features from an image
    
    Returns:
        np.array: [brightness, contrast, mean_r, mean_g, mean_b, edge_density]
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast
        contrast = np.sqrt(np.mean((gray - brightness) ** 2))
        
        # Mean RGB values
        mean_r = np.mean(img[:, :, 2])
        mean_g = np.mean(img[:, :, 1])
        mean_b = np.mean(img[:, :, 0])
        
        # Edge density
        edges = cv2.Canny(gray, canny_low_threshold, canny_high_threshold)
        edge_density = np.sum(edges) / edges.size
        
        return np.array([brightness, contrast, mean_r, mean_g, mean_b, edge_density])
    
    except Exception as e:
        print(f"Error extracting image features from {image_path}: {e}")
        return None


def extract_midi_features(midi_path, default_tempo=80):
    """
    Extract features from a MIDI file
    
    Returns:
        np.array: 20 features [tempo, avg_pitch, pitch_range, pitch_std, 
                               12 pitch class histogram values, avg_velocity, 
                               rhythmic_density, avg_duration, duration_std]
    """
    try:
        mid = mido.MidiFile(midi_path)
        microseconds_per_beat = mido.bpm2tempo(default_tempo)
        tempo_bpm = default_tempo
        ticks_per_beat = mid.ticks_per_beat
        
        # Find first tempo
        for msg in mid.tracks[0]:
            if msg.is_meta and msg.type == 'set_tempo':
                microseconds_per_beat = msg.tempo
                tempo_bpm = mido.tempo2bpm(msg.tempo)
                break
        
        pitches = []
        velocities = []
        durations = []
        note_start_times = []
        absolute_time = 0.0
        active_notes = {}
        
        for track in mid.tracks:
            absolute_time_ticks = 0
            for msg in track:
                absolute_time_ticks += msg.time
                absolute_time = mido.tick2second(absolute_time_ticks, ticks_per_beat, microseconds_per_beat)
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_key = (msg.channel, msg.note)
                    active_notes[note_key] = absolute_time
                    pitches.append(msg.note)
                    velocities.append(msg.velocity)
                    note_start_times.append(absolute_time)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    note_key = (msg.channel, msg.note)
                    if note_key in active_notes:
                        start_time = active_notes.pop(note_key)
                        duration = absolute_time - start_time
                        if duration > 1e-6:
                            durations.append(duration)
        
        # Calculate statistics
        num_notes = len(pitches)
        total_time = max(note_start_times) if note_start_times else 0.0
        
        if durations:
            last_event_time = max(s + d for s, d in zip(note_start_times[-len(durations):], durations))
            total_time = max(total_time, last_event_time)
        
        # Feature extraction
        feature_tempo = tempo_bpm
        feature_avg_pitch = np.mean(pitches) if pitches else 0
        feature_pitch_range = np.max(pitches) - np.min(pitches) if pitches else 0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feature_pitch_std = np.std(pitches) if num_notes > 1 else 0.0
        
        # Pitch class histogram
        pitch_classes = [p % 12 for p in pitches]
        pc_count = Counter(pitch_classes)
        feature_pch = np.array([pc_count.get(i, 0) for i in range(12)])
        feature_pch = feature_pch / num_notes if num_notes > 0 else feature_pch
        
        feature_avg_velocity = np.mean(velocities) if velocities else 0
        feature_rhythmic_density = (num_notes / total_time) if total_time > 0 else 0
        
        if not durations:
            feature_avg_duration = 0
            feature_duration_std = 0
        else:
            feature_avg_duration = np.mean(durations)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feature_duration_std = np.std(durations) if len(durations) > 1 else 0.0
        
        features = np.concatenate([
            np.array([feature_tempo]),
            np.array([feature_avg_pitch]),
            np.array([feature_pitch_range]),
            np.array([feature_pitch_std]),
            feature_pch,
            np.array([feature_avg_velocity]),
            np.array([feature_rhythmic_density]),
            np.array([feature_avg_duration]),
            np.array([feature_duration_std]),
        ])
        
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError("Extracted features contain NaN or Inf values.")
        
        return features
    
    except Exception as e:
        print(f"Error processing MIDI file {midi_path}: {e}")
        return None


def load_and_prepare_data(training_pairs, augment=True, augmentation_target=1000):
    """
    Load data, optionally augment it, and prepare for training
    
    Args:
        training_pairs: List of dicts with 'image_path' and 'midi_path'
        augment: Whether to apply data augmentation
        augmentation_target: Target number of pairs after augmentation
    
    Returns:
        X_data, y_data, scaler_x, scaler_y
    """
    # Augment data if requested
    if augment and len(training_pairs) < augmentation_target:
        print(f"Augmenting {len(training_pairs)} pairs to {augmentation_target} pairs...")
        augmented_pairs = run_augmentation(training_pairs, 
                                          output_dir='./augmented_data',
                                          target_total=augmentation_target)
        training_pairs = augmented_pairs
    
    # Extract features from all pairs
    X_data_list = []
    y_data_list = []
    valid_pairs_processed = 0
    
    print(f"\nExtracting features from {len(training_pairs)} pairs...")
    for pair in training_pairs:
        img_path = pair['image_path']
        midi_path = pair['midi_path']
        
        if os.path.exists(img_path) and os.path.exists(midi_path):
            image_features = extract_image_features(img_path)
            midi_features = extract_midi_features(midi_path)
            
            if image_features is not None and midi_features is not None:
                X_data_list.append(image_features)
                y_data_list.append(midi_features)
                valid_pairs_processed += 1
        else:
            print(f"Skipping pair due to missing files: {pair}")
    
    if valid_pairs_processed == 0:
        raise ValueError("No valid pairs found!")
    
    print(f"✅ Successfully processed {valid_pairs_processed} pairs")
    
    # Convert to numpy arrays
    X_data = np.array(X_data_list)
    y_data = np.array(y_data_list)
    
    # Scale features
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_data)
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y_data)
    
    return X_scaled, y_scaled, scaler_x, scaler_y


def train_model(X_train, y_train, X_val, y_val, 
                hidden_sizes=[64, 128, 128, 64],
                learning_rate=0.001,
                dropout_rate=0.3,
                epochs=2000,
                batch_size=32,
                early_stopping_patience=100):
    """
    Train the neural network model
    
    Returns:
        model, history
    """
    print("\n" + "="*50)
    print("TRAINING NEURAL NETWORK")
    print("="*50)
    
    # Initialize model
    model = ImprovedNeuralNetwork(
        input_size=6,
        hidden_sizes=hidden_sizes,
        output_size=20,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        verbose=True
    )
    
    return model, history


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Training history plot saved to: {save_path}")
    plt.close()


def save_training_artifacts(model, scaler_x, scaler_y, history, output_dir='./models'):
    """Save all training artifacts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'lofi_model.npy'
    model.save_model(str(model_path))
    
    # Save scalers
    scaler_x_path = output_dir / 'scaler_x.npy'
    scaler_y_path = output_dir / 'scaler_y.npy'
    np.save(scaler_x_path, {
        'mean': scaler_x.mean_,
        'scale': scaler_x.scale_,
        'var': scaler_x.var_
    })
    np.save(scaler_y_path, {
        'min': scaler_y.min_,
        'scale': scaler_y.scale_,
        'data_min': scaler_y.data_min_,
        'data_max': scaler_y.data_max_
    })
    
    # Save history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_json, f, indent=2)
    
    # Save training plot
    plot_path = output_dir / 'training_history.png'
    plot_training_history(history, str(plot_path))
    
    print(f"\n✅ All training artifacts saved to: {output_dir}")
    return output_dir


def run_complete_training_pipeline(original_pairs, 
                                   augment=True,
                                   augmentation_target=1000,
                                   test_size=0.15,
                                   val_size=0.15,
                                   hidden_sizes=[64, 128, 128, 64],
                                   learning_rate=0.001,
                                   dropout_rate=0.3,
                                   epochs=2000,
                                   batch_size=32,
                                   early_stopping_patience=100,
                                   output_dir='./models'):
    """
    Complete training pipeline from data loading to model saving
    
    Args:
        original_pairs: List of dicts with 'image_path' and 'midi_path'
        augment: Whether to augment data
        augmentation_target: Target number of pairs after augmentation
        test_size: Fraction of data for testing
        val_size: Fraction of data for validation
        hidden_sizes: List of hidden layer sizes
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        epochs: Maximum number of training epochs
        batch_size: Mini-batch size
        early_stopping_patience: Early stopping patience
        output_dir: Directory to save model and artifacts
    
    Returns:
        model, scaler_x, scaler_y, history
    """
    print("="*60)
    print("LOFI GENERATOR - COMPLETE TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Load and prepare data
    print("\n📦 STEP 1: Loading and preparing data...")
    X_data, y_data, scaler_x, scaler_y = load_and_prepare_data(
        original_pairs, 
        augment=augment,
        augmentation_target=augmentation_target
    )
    
    # Step 2: Split data
    print(f"\n✂️ STEP 2: Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train model
    print(f"\n🚀 STEP 3: Training model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience
    )
    
    # Step 4: Evaluate on test set
    print(f"\n📊 STEP 4: Evaluating on test set...")
    test_predictions = model.predict(X_test)
    test_loss = model.mean_squared_error(y_test, test_predictions)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    
    # Step 5: Save everything
    print(f"\n💾 STEP 5: Saving model and artifacts...")
    save_training_artifacts(model, scaler_x, scaler_y, history, output_dir)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  - Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"  - Test loss: {test_loss:.6f}")
    print(f"  - Model saved to: {output_dir}")
    
    return model, scaler_x, scaler_y, history


if __name__ == "__main__":
    print("Training Pipeline Ready!")
    print("\nExample usage:")
    print("""
    from training_pipeline import run_complete_training_pipeline
    
    original_pairs = [
        {'image_path': 'path/to/image1.jpg', 'midi_path': 'path/to/midi1.mid'},
        {'image_path': 'path/to/image2.jpg', 'midi_path': 'path/to/midi2.mid'},
        # ... add all 50 pairs
    ]
    
    model, scaler_x, scaler_y, history = run_complete_training_pipeline(
        original_pairs,
        augment=True,
        augmentation_target=1000,
        epochs=2000
    )
    """)
