"""
MIDI to MP3 Conversion
Converts generated MIDI files to MP3 audio using FluidSynth
"""

from midi2audio import FluidSynth
import os
from pathlib import Path

# Default soundfont path (adjust based on your system)
# On Mac with Homebrew: /opt/homebrew/share/soundfonts/default.sf2
# On Linux: /usr/share/sounds/sf2/FluidR3_GM.sf2
SOUNDFONT_PATH = "/opt/homebrew/share/soundfonts/default.sf2"

# Alternative: You can download a custom lofi soundfont for better results
# https://musical-artifacts.com/artifacts?tags=soundfont


def find_soundfont():
    """Find available soundfont on the system"""
    possible_paths = [
        "/opt/homebrew/share/soundfonts/default.sf2",
        "/usr/local/share/soundfonts/default.sf2",
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default.sf2",
        "./soundfonts/default.sf2",  # Local fallback
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found soundfont: {path}")
            return path
    
    print("⚠️ No soundfont found! Audio quality may be affected.")
    return None


def convert_midi_to_mp3(midi_path, mp3_path, soundfont_path=None):
    """
    Convert MIDI file to MP3
    
    Args:
        midi_path: Path to input MIDI file
        mp3_path: Path to output MP3 file
        soundfont_path: Path to soundfont file (optional)
    
    Returns:
        bool: Success status
    """
    try:
        # Find soundfont if not provided
        if soundfont_path is None:
            soundfont_path = find_soundfont()
        
        if soundfont_path is None:
            print("❌ Cannot convert without soundfont")
            return False
        
        # Check if MIDI file exists
        if not os.path.exists(midi_path):
            print(f"❌ MIDI file not found: {midi_path}")
            return False
        
        print(f"🎵 Converting MIDI to MP3...")
        print(f"   Input: {midi_path}")
        print(f"   Output: {mp3_path}")
        
        # Initialize FluidSynth with soundfont
        fs = FluidSynth(soundfont_path)
        
        # Convert MIDI to WAV first (intermediate step)
        wav_path = mp3_path.replace('.mp3', '.wav')
        fs.midi_to_audio(midi_path, wav_path)
        
        # Convert WAV to MP3 using FFmpeg (if available)
        try:
            import subprocess
            result = subprocess.run(
                ['ffmpeg', '-i', wav_path, '-b:a', '192k', mp3_path, '-y'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ MP3 created: {mp3_path}")
                # Clean up WAV file
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                return True
            else:
                print(f"⚠️ FFmpeg conversion failed, keeping WAV file")
                # Rename WAV to MP3 as fallback
                os.rename(wav_path, mp3_path.replace('.mp3', '.wav'))
                return True
                
        except FileNotFoundError:
            print("⚠️ FFmpeg not found, keeping WAV file")
            # Rename WAV to keep it
            os.rename(wav_path, mp3_path.replace('.mp3', '.wav'))
            return True
    
    except Exception as e:
        print(f"❌ Error converting MIDI to MP3: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_midi_to_wav(midi_path, wav_path, soundfont_path=None):
    """
    Convert MIDI file to WAV (simpler, no FFmpeg required)
    
    Args:
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
        soundfont_path: Path to soundfont file (optional)
    
    Returns:
        bool: Success status
    """
    try:
        # Find soundfont if not provided
        if soundfont_path is None:
            soundfont_path = find_soundfont()
        
        if soundfont_path is None:
            print("❌ Cannot convert without soundfont")
            return False
        
        if not os.path.exists(midi_path):
            print(f"❌ MIDI file not found: {midi_path}")
            return False
        
        print(f"🎵 Converting MIDI to WAV...")
        print(f"   Input: {midi_path}")
        print(f"   Output: {wav_path}")
        
        # Initialize FluidSynth with soundfont
        fs = FluidSynth(soundfont_path)
        
        # Convert
        fs.midi_to_audio(midi_path, wav_path)
        
        print(f"✅ WAV created: {wav_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error converting MIDI to WAV: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the conversion
    print("Testing MIDI to audio conversion...")
    soundfont = find_soundfont()
    if soundfont:
        print(f"Soundfont found: {soundfont}")
    else:
        print("No soundfont found. Install with: brew install fluid-synth")
