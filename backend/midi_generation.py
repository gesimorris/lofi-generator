"""
MIDI Generation Module
Generates MIDI files from predicted music parameters using simulated annealing optimization
"""

import numpy as np
import mido
import random
import math
from collections import Counter


# Constants
MIDI_PITCH_RANGE_CLAMP = (36, 96)
MIN_NOTE_DURATION_SEC = 0.1
MAX_NOTE_DURATION_SEC = 2.0
C_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
A_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
TARGET_SCALE = C_MAJOR_SCALE

# Fitness weights
W_SCALE = 0.3
W_INTERVAL = 0.25
W_RYTHMIC = 0.15
W_PITCH_RANGE = 0.1
W_PITCH_STD = 0.1
W_DURATION = 0.1


def scale_output_to_music(scaled_prediction, scaler_y):
    """
    Convert scaled neural network output to music parameters
    
    Args:
        scaled_prediction: Scaled prediction from neural network
        scaler_y: MinMaxScaler used during training
    
    Returns:
        dict: Music parameters
    """
    try:
        # Ensure input is 2D for the scaler
        if scaled_prediction.ndim == 1:
            scaled_prediction = scaled_prediction.reshape(1, -1)
        
        if scaled_prediction.shape[1] != scaler_y.n_features_in_:
            raise ValueError(f"Prediction has {scaled_prediction.shape[1]} features, "
                           f"but scaler_y expects {scaler_y.n_features_in_}")
        
        # Inverse transform
        inverse_scaled = scaler_y.inverse_transform(scaled_prediction)
        pred = inverse_scaled.flatten()
        
        # Extract and clamp values
        tempo = int(round(max(50, min(pred[0], 160))))
        avg_pitch = int(round(max(36, min(pred[1], 96))))
        pitch_range = max(0, min(pred[2], 48))
        pitch_std = max(0, min(pred[3], 12))
        
        pch = pred[4:16]
        pch[pch < 0] = 0
        pch_sum = np.sum(pch)
        if pch_sum > 1e-6:
            pch = pch / pch_sum
        else:
            pch = np.ones(12) / 12.0
        
        avg_velocity = int(round(max(30, min(pred[16], 100))))
        rhythmic_density = max(0.1, min(pred[17], 10))
        avg_duration = max(0.1, min(pred[18], 4.0))
        duration_std = max(0, min(pred[19], 2.0))
        
        music_params = {
            'tempo': tempo,
            'average_pitch': avg_pitch,
            'pitch_range': pitch_range,
            'pitch_std': pitch_std,
            'pitch_class_histogram': pch,
            'average_velocity': avg_velocity,
            'rhythmic_density': rhythmic_density,
            'average_duration': avg_duration,
            'duration_std': duration_std
        }
        
        return music_params
    
    except Exception as e:
        print(f"Error during inverse scaling: {e}")
        return None


def generate_initial_sequence(music_params, target_duration=15):
    """
    Generate initial melody sequence from music parameters
    
    Args:
        music_params: Dictionary of music parameters
        target_duration: Target duration in seconds
    
    Returns:
        list: List of note dictionaries with pitch, duration, velocity
    """
    avg_pitch = music_params.get('average_pitch', 60)
    avg_velocity = music_params.get('average_velocity', 70)
    avg_duration = music_params.get('average_duration', 0.5)
    rhythmic_density = music_params.get('rhythmic_density', 2)
    duration_std = music_params.get('duration_std', 0.1)
    pitch_std = music_params.get('pitch_std', 5)
    pch = music_params.get('pitch_class_histogram', np.ones(12) / 12.0)
    
    generated_notes = []
    target_num_notes = max(2, int(rhythmic_density * target_duration))
    last_pitch = int(round(avg_pitch))
    
    for i in range(target_num_notes):
        # Choose pitch class from histogram
        pitch_class = np.random.choice(12, p=pch)
        
        # Determine pitch near last pitch
        pitch_candidate_low = last_pitch - int(round(pitch_std * random.uniform(0.5, 1.5)))
        pitch_candidate_high = last_pitch + int(round(pitch_std * random.uniform(0.5, 1.5)))
        target_pitch_area = random.uniform(pitch_candidate_low, pitch_candidate_high)
        
        # Find closest pitch with chosen pitch class
        octave_base = int(round(target_pitch_area / 12.0)) * 12
        pitch = octave_base + pitch_class
        
        # Try adjacent octaves if too far
        if abs(pitch - target_pitch_area) > 6:
            pitch_alt1 = pitch - 12
            pitch_alt2 = pitch + 12
            if abs(pitch_alt1 - target_pitch_area) < abs(pitch - target_pitch_area):
                pitch = pitch_alt1
            elif abs(pitch_alt2 - target_pitch_area) < abs(pitch - target_pitch_area):
                pitch = pitch_alt2
        
        # Clamp pitch
        pitch = int(round(max(MIDI_PITCH_RANGE_CLAMP[0], 
                             min(pitch, MIDI_PITCH_RANGE_CLAMP[1]))))
        last_pitch = pitch
        
        # Choose duration
        duration_sec = np.random.normal(loc=avg_duration, scale=max(0.01, duration_std))
        duration_sec = max(MIN_NOTE_DURATION_SEC, min(duration_sec, MAX_NOTE_DURATION_SEC))
        
        # Set velocity
        velocity = int(round(avg_velocity))
        
        generated_notes.append({
            'pitch': pitch,
            'duration': duration_sec,
            'velocity': velocity
        })
    
    return generated_notes


def calculate_fitness(melody, music_params):
    """
    Calculate fitness score for a melody
    
    Args:
        melody: List of note dictionaries
        music_params: Target music parameters
    
    Returns:
        float: Fitness score (higher is better)
    """
    pitches = np.array([note['pitch'] for note in melody])
    durations = np.array([note['duration'] for note in melody])
    num_notes = len(melody)
    
    # Scale fitness
    notes_in_scale = sum(1 for pitch in pitches if (pitch % 12) in TARGET_SCALE)
    scale_fitness = notes_in_scale / num_notes if num_notes > 0 else 0
    
    # Interval fitness (penalize large leaps)
    intervals = np.abs(np.diff(pitches))
    large_leap_penalty = np.mean(np.maximum(0, intervals - 12)**2)
    interval_fitness = 1.0 / (1.0 + large_leap_penalty + 1e-6)
    
    # Rhythmic density fitness
    target_dur_std = music_params.get('duration_std', 0.1)
    actual_dur_std = np.std(durations) if num_notes > 0 else 0
    if target_dur_std < 1e-6:
        rhythmic_density_fitness = 1.0 if actual_dur_std < 1e-6 else 0.0
    else:
        rhythmic_density_fitness = 1.0 / (1.0 + np.abs(actual_dur_std - target_dur_std) + 1e-6)
    
    # Pitch range fitness
    target_pitch_range = music_params.get('pitch_range', 12)
    actual_pitch_range = np.max(pitches) - np.min(pitches) if num_notes > 0 else 0
    range_different_penalty = abs(actual_pitch_range - target_pitch_range) / (target_pitch_range + 1e-6)
    pitch_range_fitness = max(0, 1.0 - range_different_penalty)
    
    # Pitch std fitness
    target_pitch_std = music_params.get('pitch_std', 5)
    actual_pitch_std = np.std(pitches) if num_notes > 1 else 0
    pitch_std_score = 1.0 / (1.0 + np.abs(actual_pitch_std - target_pitch_std) + 1e-6)
    
    # Average duration fitness
    target_avg_duration = music_params.get('average_duration', 0.5)
    actual_avg_duration = np.mean(durations) if num_notes > 0 else 0
    avg_duration_score = 1.0 / (1.0 + np.abs(actual_avg_duration - target_avg_duration) + 1e-6)
    
    # Total fitness
    total_fitness = (
        W_SCALE * scale_fitness +
        W_INTERVAL * interval_fitness +
        W_RYTHMIC * rhythmic_density_fitness +
        W_PITCH_RANGE * pitch_range_fitness +
        W_PITCH_STD * pitch_std_score +
        W_DURATION * avg_duration_score
    )
    
    return total_fitness


def get_neighbor(melody):
    """
    Generate a neighbor melody by mutating one note
    
    Args:
        melody: List of note dictionaries
    
    Returns:
        list: Modified melody
    """
    neighbor_melody = [note.copy() for note in melody]
    chosen_index = random.randrange(len(neighbor_melody))
    
    mutation_type = random.choice(['pitch', 'duration', 'velocity'])
    
    if mutation_type == 'pitch':
        pitch_change = random.choice([-3, -2, -1, 1, 2, 3])
        new_pitch = neighbor_melody[chosen_index]['pitch'] + pitch_change
        neighbor_melody[chosen_index]['pitch'] = int(round(
            max(MIDI_PITCH_RANGE_CLAMP[0], min(new_pitch, MIDI_PITCH_RANGE_CLAMP[1]))
        ))
    
    elif mutation_type == 'duration':
        duration_change = random.uniform(0.7, 1.3)
        new_duration = neighbor_melody[chosen_index]['duration'] * duration_change
        neighbor_melody[chosen_index]['duration'] = max(
            MIN_NOTE_DURATION_SEC, min(new_duration, MAX_NOTE_DURATION_SEC)
        )
    
    elif mutation_type == 'velocity':
        velocity_change = random.choice([-20, -10, 10, 20])
        new_velocity = neighbor_melody[chosen_index]['velocity'] + velocity_change
        neighbor_melody[chosen_index]['velocity'] = max(0, min(new_velocity, 127))
    
    return neighbor_melody


def simulated_annealing(melody, music_params, initial_temp=1.0, 
                       cooling_rate=0.995, max_iterations=3000):
    """
    Optimize melody using simulated annealing
    
    Args:
        melody: Initial melody
        music_params: Target music parameters
        initial_temp: Initial temperature
        cooling_rate: Cooling rate per iteration
        max_iterations: Maximum iterations
    
    Returns:
        list: Optimized melody
    """
    current_melody = melody
    current_fitness = calculate_fitness(current_melody, music_params)
    best_melody = current_melody
    best_fitness = current_fitness
    temp = initial_temp
    
    for iteration in range(max_iterations):
        neighbor_melody = get_neighbor(current_melody)
        neighbor_fitness = calculate_fitness(neighbor_melody, music_params)
        
        fitness_change = neighbor_fitness - current_fitness
        
        if fitness_change > 0:
            current_melody = neighbor_melody
            current_fitness = neighbor_fitness
        else:
            if temp > 1e-6:
                acceptance_probability = math.exp(fitness_change / temp)
                if random.random() < acceptance_probability:
                    current_melody = neighbor_melody
                    current_fitness = neighbor_fitness
        
        if current_fitness > best_fitness:
            best_melody = current_melody
            best_fitness = current_fitness
        
        temp *= cooling_rate
    
    return best_melody


def write_melody_to_midi(sequence, music_params, filename):
    """
    Write melody sequence to MIDI file
    
    Args:
        sequence: List of note dictionaries
        music_params: Music parameters (for tempo)
        filename: Output MIDI filename
    """
    try:
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        tempo = music_params.get('tempo', 80)
        ticks_per_beat = mid.ticks_per_beat
        microseconds_per_beat = mido.bpm2tempo(tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
        
        absolute_time_ticks = 0
        
        for note in sequence:
            pitch = note.get('pitch', 60)
            duration_sec = note.get('duration', 0.5)
            velocity = note.get('velocity', 70)
            
            start_delta_ticks = 5 if absolute_time_ticks > 0 else 0
            absolute_time_ticks += start_delta_ticks
            
            duration_ticks = int(mido.second2tick(duration_sec, ticks_per_beat, microseconds_per_beat))
            duration_ticks = max(1, duration_ticks)
            
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=start_delta_ticks))
            track.append(mido.Message('note_off', note=pitch, velocity=velocity, time=duration_ticks))
            
            absolute_time_ticks += duration_ticks
        
        track.append(mido.MetaMessage('end_of_track', time=ticks_per_beat))
        mid.save(filename)
        print(f"✅ MIDI file saved: {filename}")
        return True
    
    except Exception as e:
        print(f"❌ Error creating MIDI file '{filename}': {e}")
        return False


def generate_music_from_prediction(predicted_music, scaler_y, output_filename,
                                   sa_iterations=3000, sa_temp=0.5, sa_cool=0.997,
                                   target_duration=15):
    """
    Generate complete MIDI file from neural network prediction
    
    Args:
        predicted_music: Raw prediction from neural network
        scaler_y: Scaler to inverse transform prediction
        output_filename: Output MIDI filename
        sa_iterations: Simulated annealing iterations
        sa_temp: Initial temperature
        sa_cool: Cooling rate
        target_duration: Target duration in seconds
    
    Returns:
        bool: Success status
    """
    try:
        # Convert prediction to music parameters
        music_params = scale_output_to_music(predicted_music, scaler_y)
        if music_params is None:
            return False
        
        print("\n🎵 Predicted music parameters:")
        for key, value in music_params.items():
            if key != 'pitch_class_histogram':
                print(f"  {key}: {value}")
        
        # Generate initial melody
        print("\n🎼 Generating initial melody...")
        initial_melody = generate_initial_sequence(music_params, target_duration=target_duration)
        initial_fitness = calculate_fitness(initial_melody, music_params)
        print(f"Initial fitness: {initial_fitness:.4f}")
        
        # Optimize with simulated annealing
        print(f"\n🔥 Optimizing with simulated annealing ({sa_iterations} iterations)...")
        refined_melody = simulated_annealing(
            initial_melody, music_params,
            max_iterations=sa_iterations,
            initial_temp=sa_temp,
            cooling_rate=sa_cool
        )
        refined_fitness = calculate_fitness(refined_melody, music_params)
        print(f"Refined fitness: {refined_fitness:.4f}")
        print(f"Improvement: {((refined_fitness - initial_fitness) / initial_fitness * 100):.2f}%")
        
        # Save to MIDI
        print(f"\n💾 Saving MIDI file...")
        success = write_melody_to_midi(refined_melody, music_params, output_filename)
        
        return success
    
    except Exception as e:
        print(f"❌ Error generating music: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("MIDI Generation Module Ready!")
