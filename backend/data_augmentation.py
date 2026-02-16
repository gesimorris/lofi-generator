"""
Data Augmentation Script
Expands 50 image-MIDI pairs to 1000+ pairs through various transformations
"""

import cv2
import numpy as np
import os
import mido
from pathlib import Path
import json
from tqdm import tqdm

class ImageAugmentor:
    """Apply various transformations to images"""
    
    @staticmethod
    def adjust_brightness(image, factor):
        """Adjust brightness: factor > 1 brightens, < 1 darkens"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_contrast(image, factor):
        """Adjust contrast: factor > 1 increases contrast"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        adjusted = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255)
        return adjusted.astype(np.uint8)
    
    @staticmethod
    def adjust_saturation(image, factor):
        """Adjust saturation: factor > 1 increases saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_hue(image, shift):
        """Shift hue: shift in range [-180, 180]"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def add_noise(image, noise_level=0.05):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    @staticmethod
    def blur(image, kernel_size=5):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def rotate(image, angle):
        """Rotate image by angle in degrees"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), 
                             borderMode=cv2.BORDER_REFLECT)
    
    @staticmethod
    def flip(image, mode='horizontal'):
        """Flip image horizontally or vertically"""
        if mode == 'horizontal':
            return cv2.flip(image, 1)
        elif mode == 'vertical':
            return cv2.flip(image, 0)
        return image
    
    @staticmethod
    def crop_and_resize(image, crop_factor=0.9):
        """Crop center and resize back to original size"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
        cropped = image[start_h:start_h + new_h, start_w:start_w + new_w]
        return cv2.resize(cropped, (w, h))


class MIDIAugmentor:
    """Apply various transformations to MIDI files"""
    
    @staticmethod
    def transpose(midi_path, semitones, output_path):
        """Transpose MIDI by semitones"""
        try:
            mid = mido.MidiFile(midi_path)
            new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
            
            for track in mid.tracks:
                new_track = mido.MidiTrack()
                for msg in track:
                    if msg.type in ['note_on', 'note_off']:
                        new_note = max(0, min(127, msg.note + semitones))
                        new_msg = msg.copy(note=new_note)
                        new_track.append(new_msg)
                    else:
                        new_track.append(msg.copy())
                new_mid.tracks.append(new_track)
            
            new_mid.save(output_path)
            return True
        except Exception as e:
            print(f"Error transposing MIDI: {e}")
            return False
    
    @staticmethod
    def change_tempo(midi_path, tempo_factor, output_path):
        """Change tempo by a factor (1.0 = no change, 1.2 = 20% faster)"""
        try:
            mid = mido.MidiFile(midi_path)
            new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
            
            for track in mid.tracks:
                new_track = mido.MidiTrack()
                for msg in track:
                    if msg.is_meta and msg.type == 'set_tempo':
                        new_tempo = int(msg.tempo / tempo_factor)
                        new_msg = mido.MetaMessage('set_tempo', tempo=new_tempo, time=msg.time)
                        new_track.append(new_msg)
                    else:
                        new_track.append(msg.copy())
                new_mid.tracks.append(new_track)
            
            new_mid.save(output_path)
            return True
        except Exception as e:
            print(f"Error changing tempo: {e}")
            return False
    
    @staticmethod
    def adjust_velocity(midi_path, velocity_factor, output_path):
        """Adjust note velocities by a factor"""
        try:
            mid = mido.MidiFile(midi_path)
            new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
            
            for track in mid.tracks:
                new_track = mido.MidiTrack()
                for msg in track:
                    if msg.type == 'note_on':
                        new_velocity = max(1, min(127, int(msg.velocity * velocity_factor)))
                        new_msg = msg.copy(velocity=new_velocity)
                        new_track.append(new_msg)
                    else:
                        new_track.append(msg.copy())
                new_mid.tracks.append(new_track)
            
            new_mid.save(output_path)
            return True
        except Exception as e:
            print(f"Error adjusting velocity: {e}")
            return False


class DataAugmentationPipeline:
    """Complete data augmentation pipeline"""
    
    def __init__(self, original_pairs, output_dir, target_total=1000):
        """
        Args:
            original_pairs: List of dicts with 'image_path' and 'midi_path'
            output_dir: Directory to save augmented data
            target_total: Target number of pairs to generate
        """
        self.original_pairs = original_pairs
        self.output_dir = Path(output_dir)
        self.target_total = target_total
        
        # Create output directories
        self.images_dir = self.output_dir / 'images'
        self.midi_dir = self.output_dir / 'midi'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.midi_dir.mkdir(parents=True, exist_ok=True)
        
        self.img_aug = ImageAugmentor()
        self.midi_aug = MIDIAugmentor()
    
    def generate_augmentation_configs(self):
        """Generate various augmentation configurations"""
        configs = []
        
        # Brightness variations
        for factor in [0.7, 0.85, 1.15, 1.3]:
            configs.append({
                'name': f'brightness_{factor}',
                'image_fn': lambda img, f=factor: self.img_aug.adjust_brightness(img, f),
                'midi_fn': None
            })
        
        # Contrast variations
        for factor in [0.8, 1.2, 1.4]:
            configs.append({
                'name': f'contrast_{factor}',
                'image_fn': lambda img, f=factor: self.img_aug.adjust_contrast(img, f),
                'midi_fn': None
            })
        
        # Saturation variations
        for factor in [0.6, 1.3, 1.6]:
            configs.append({
                'name': f'saturation_{factor}',
                'image_fn': lambda img, f=factor: self.img_aug.adjust_saturation(img, f),
                'midi_fn': lambda path, out, f=factor: self.midi_aug.adjust_velocity(path, f * 0.8, out)
            })
        
        # Hue shifts
        for shift in [-30, -15, 15, 30]:
            configs.append({
                'name': f'hue_{shift}',
                'image_fn': lambda img, s=shift: self.img_aug.adjust_hue(img, s),
                'midi_fn': lambda path, out, s=shift: self.midi_aug.transpose(path, s // 15, out)
            })
        
        # Combined transformations
        configs.append({
            'name': 'warm_bright',
            'image_fn': lambda img: self.img_aug.adjust_hue(
                self.img_aug.adjust_brightness(img, 1.2), 10),
            'midi_fn': lambda path, out: self.midi_aug.transpose(path, 2, out)
        })
        
        configs.append({
            'name': 'cool_dark',
            'image_fn': lambda img: self.img_aug.adjust_hue(
                self.img_aug.adjust_brightness(img, 0.8), -10),
            'midi_fn': lambda path, out: self.midi_aug.transpose(path, -2, out)
        })
        
        configs.append({
            'name': 'vibrant',
            'image_fn': lambda img: self.img_aug.adjust_saturation(
                self.img_aug.adjust_contrast(img, 1.3), 1.4),
            'midi_fn': lambda path, out: self.midi_aug.adjust_velocity(path, 1.2, out)
        })
        
        configs.append({
            'name': 'muted',
            'image_fn': lambda img: self.img_aug.adjust_saturation(
                self.img_aug.adjust_contrast(img, 0.9), 0.7),
            'midi_fn': lambda path, out: self.midi_aug.adjust_velocity(path, 0.8, out)
        })
        
        # Noise and blur
        configs.append({
            'name': 'noise',
            'image_fn': lambda img: self.img_aug.add_noise(img, 0.03),
            'midi_fn': None
        })
        
        configs.append({
            'name': 'blur',
            'image_fn': lambda img: self.img_aug.blur(img, 5),
            'midi_fn': None
        })
        
        # Geometric transformations
        for angle in [-15, -10, 10, 15]:
            configs.append({
                'name': f'rotate_{angle}',
                'image_fn': lambda img, a=angle: self.img_aug.rotate(img, a),
                'midi_fn': None
            })
        
        configs.append({
            'name': 'flip_h',
            'image_fn': lambda img: self.img_aug.flip(img, 'horizontal'),
            'midi_fn': None
        })
        
        configs.append({
            'name': 'crop',
            'image_fn': lambda img: self.img_aug.crop_and_resize(img, 0.9),
            'midi_fn': None
        })
        
        return configs
    
    def augment_dataset(self):
        """Generate augmented dataset"""
        configs = self.generate_augmentation_configs()
        num_original = len(self.original_pairs)
        augmentations_per_pair = (self.target_total - num_original) // num_original
        
        augmented_pairs = []
        
        print(f"Generating {self.target_total} pairs from {num_original} originals...")
        print(f"Creating ~{augmentations_per_pair} augmentations per original pair")
        
        # First, copy original pairs
        for idx, pair in enumerate(tqdm(self.original_pairs, desc="Copying originals")):
            img_path = pair['image_path']
            midi_path = pair['midi_path']
            
            if not os.path.exists(img_path) or not os.path.exists(midi_path):
                print(f"Skipping missing pair: {pair}")
                continue
            
            # Copy original image
            img = cv2.imread(img_path)
            new_img_path = self.images_dir / f"original_{idx:04d}.jpg"
            cv2.imwrite(str(new_img_path), img)
            
            # Copy original MIDI
            new_midi_path = self.midi_dir / f"original_{idx:04d}.mid"
            mid = mido.MidiFile(midi_path)
            mid.save(str(new_midi_path))
            
            augmented_pairs.append({
                'image_path': str(new_img_path),
                'midi_path': str(new_midi_path),
                'augmentation': 'original'
            })
        
        # Generate augmentations
        for idx, pair in enumerate(tqdm(self.original_pairs, desc="Generating augmentations")):
            img_path = pair['image_path']
            midi_path = pair['midi_path']
            
            if not os.path.exists(img_path) or not os.path.exists(midi_path):
                continue
            
            img = cv2.imread(img_path)
            
            # Apply augmentations
            for config_idx, config in enumerate(configs[:augmentations_per_pair]):
                try:
                    # Augment image
                    aug_img = config['image_fn'](img)
                    aug_img_path = self.images_dir / f"aug_{idx:04d}_{config['name']}.jpg"
                    cv2.imwrite(str(aug_img_path), aug_img)
                    
                    # Augment MIDI if function provided
                    aug_midi_path = self.midi_dir / f"aug_{idx:04d}_{config['name']}.mid"
                    if config['midi_fn'] is not None:
                        success = config['midi_fn'](midi_path, str(aug_midi_path))
                        if not success:
                            # If augmentation fails, copy original
                            mid = mido.MidiFile(midi_path)
                            mid.save(str(aug_midi_path))
                    else:
                        # Copy original MIDI
                        mid = mido.MidiFile(midi_path)
                        mid.save(str(aug_midi_path))
                    
                    augmented_pairs.append({
                        'image_path': str(aug_img_path),
                        'midi_path': str(aug_midi_path),
                        'augmentation': config['name']
                    })
                
                except Exception as e:
                    print(f"Error augmenting pair {idx} with {config['name']}: {e}")
                    continue
        
        # Save augmented pairs metadata
        metadata_path = self.output_dir / 'augmented_pairs.json'
        with open(metadata_path, 'w') as f:
            json.dump(augmented_pairs, f, indent=2)
        
        print(f"\n✅ Generated {len(augmented_pairs)} pairs!")
        print(f"📁 Images saved to: {self.images_dir}")
        print(f"🎵 MIDI files saved to: {self.midi_dir}")
        print(f"📋 Metadata saved to: {metadata_path}")
        
        return augmented_pairs


def run_augmentation(original_pairs, output_dir='./augmented_data', target_total=1000):
    """
    Main function to run data augmentation
    
    Args:
        original_pairs: List of dicts with 'image_path' and 'midi_path'
        output_dir: Output directory for augmented data
        target_total: Target number of pairs to generate
    
    Returns:
        List of augmented pairs
    """
    pipeline = DataAugmentationPipeline(original_pairs, output_dir, target_total)
    return pipeline.augment_dataset()


if __name__ == "__main__":
    # Example usage
    print("Data Augmentation Script Ready!")
    print("\nExample usage:")
    print("""
    from data_augmentation import run_augmentation
    
    original_pairs = [
        {'image_path': 'path/to/image1.jpg', 'midi_path': 'path/to/midi1.mid'},
        {'image_path': 'path/to/image2.jpg', 'midi_path': 'path/to/midi2.mid'},
        # ... more pairs
    ]
    
    augmented_pairs = run_augmentation(original_pairs, 
                                       output_dir='./augmented_data',
                                       target_total=1000)
    """)
