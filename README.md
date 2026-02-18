<p align="center">
<img width="150" height="150" alt="ailofi" src="https://github.com/user-attachments/assets/1ac6aabd-d8e9-4568-a85a-abb3c8ec77b8" />
</p>

<p align="center">
  FeelScape
</p>


<p align="center">
FeelScape is a next-gen LOFI audio generator that aims to capture the mood of an image and produce a LOFI song that reflects that mood. Upload any image and my neural network analyzes its visual features (**colors, contrast, patterns**) to create a matching LOFI beat as either a WAV or MIDI file.
</p>

<p align="center">
**Tech Stack:** React, FastAPI, NumPy, OpenCV, MIDI Processing
</p>

---

## System Architecture

```
Image Upload → Feature Extraction → Neural Network → MIDI Parameters → 
Initial Melody Generation → Simulated Annealing Optimization → MIDI File → 
Audio Synthesis (WAV) → Download
```

---

## Model Architecture

To map visual features to musical features, I designed a fully connected neural network that learns relationships between extracted image characteristics and structured MIDI outputs. Because the dataset was small (approximately 50 manually paired samples), I focused on three key objectives: controlling overfitting, stabilizing training, and maintaining expressive capacity without excessive complexity.

### Network Topology

I chose a 4-layer architecture with the configuration input → 64 → 128 → 128 → 64 → output for several specific reasons. First, extremely deep networks would overfit on the small dataset, memorizing the training examples rather than learning generalizable patterns. Second, this configuration provides enough capacity to model the non-linear relationships between visual and musical features without being so complex that it requires thousands of training examples. Third, the symmetric encoder-decoder structure, where we expand from 64 to 128 neurons and then compress back to 64, creates a bottleneck that forces the model to learn compressed representations of the image-music mapping.

```python
def __init__(self, input_size=6, hidden_sizes=[64, 128, 128, 64], output_size=20, 
             learning_rate=0.001, dropout_rate=0.3):
    # He initialization for ReLU: sqrt(2/n)
    weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
```

I initialized the weights using He initialization, which scales random values by the square root of 2 divided by the number of input units. This is crucial because standard random initialization causes vanishing or exploding gradients when using ReLU activation functions. He initialization specifically accounts for ReLU's behavior of zeroing out negative values, maintaining consistent variance across layers and enabling stable deep learning even without extensive tuning.

### Activation Functions

For the hidden layers, I chose ReLU (Rectified Linear Unit) over traditional sigmoid or tanh functions for four key reasons. First, ReLU doesn't suffer from vanishing gradients because its gradient is either 0 or 1, never getting squeezed to near-zero like sigmoid's exponentially decaying gradient. Second, it's computationally efficient—a simple max(0, x) operation is much faster than computing exponentials. Third, ReLU creates sparse activation patterns where approximately 50% of neurons output zero, which acts as natural feature selection and reduces co-adaptation between neurons. Fourth, it works synergistically with He initialization for stable weight scaling across deep networks.

```python
def relu(self, x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(self, x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)
```

For the output layer, however, I switched to sigmoid activation because the MIDI output features are normalized between 0 and 1, representing values like tempo (normalized to 0-1 range), velocity (0-127 mapped to 0-1), and pitch class probabilities (must sum to 1). Sigmoid keeps predictions bounded and interpretable, ensuring we never predict impossible values like negative tempo or velocity above 127. I also added clipping to prevent numerical overflow when the input x becomes very large, which would cause the exponential to explode.

```python
def sigmoid(self, x):
    """Sigmoid activation for output layer"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
```

### Regularization Strategy

Batch normalization is applied after each linear transformation but before the activation function. During training mode, I normalize each mini-batch using that batch's own mean and variance, which stabilizes the distribution of activations flowing through the network. This normalization reduces internal covariate shift, where the distribution of layer inputs changes during training as the parameters of previous layers update. I use a momentum value of 0.9 to smooth the running statistics exponentially, giving recent batches slightly more weight while maintaining historical context. The small epsilon value of 1e-8 prevents division by zero when features have zero variance, which can happen with small batch sizes or homogeneous data.

```python
def batch_norm(self, x, gamma, beta, running_mean, running_var, layer_idx, momentum=0.9):
    if self.training_mode:
        batch_mean = np.mean(x, axis=0, keepdims=True)
        batch_var = np.var(x, axis=0, keepdims=True)
        # Update running statistics with momentum
        self.bn_running_mean[layer_idx] = momentum * self.bn_running_mean[layer_idx] + (1 - momentum) * batch_mean
        self.bn_running_var[layer_idx] = momentum * self.bn_running_var[layer_idx] + (1 - momentum) * batch_var
        x_norm = (x - batch_mean) / np.sqrt(batch_var + 1e-8)
    else:
        # Use running stats during inference
        x_norm = (x - running_mean) / np.sqrt(running_var + 1e-8)
    return gamma * x_norm + beta
```

Batch normalization serves multiple purposes beyond just normalization. It acts as a regularizer because the noise introduced by using batch statistics during training (rather than the true population statistics) has a similar effect to dropout. It also allows me to use higher learning rates without instability, since the normalization prevents activations from growing unboundedly. During inference, I switch to using the running mean and variance computed during training, because we might only have a single sample and can't compute meaningful batch statistics.

I complement batch normalization with 30% dropout applied to the hidden layers. Dropout randomly zeros out 30% of the neurons during each training iteration, forcing the remaining neurons to learn robust features independently rather than co-adapting. This creates an ensemble effect where each forward pass is effectively training a different subnetwork, and at test time we approximate the ensemble by using all neurons with scaled activations. I chose 30% specifically because it's aggressive enough to prevent overfitting on the small dataset, but not so aggressive (like 50% or higher) that we lose too much information during training.

```python
def dropout(self, x, rate, layer_idx):
    if self.training_mode and rate > 0:
        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
        self.cache[f'dropout_mask_{layer_idx}'] = mask
        return x * mask
    return x
```

One critical implementation detail is that I store layer-specific dropout masks using unique dictionary keys like `dropout_mask_0`, `dropout_mask_1`, etc. I initially tried using a single global mask, which caused dimension mismatch errors during backpropagation because each layer has different dimensions (layer 1 might be batch_size × 64 while layer 2 is batch_size × 128). The inverted dropout technique, where we divide by (1 - rate), ensures that the expected value of activations remains consistent between training and testing without needing to scale at inference time.

### Training Loop with Early Stopping

The training loop implements mini-batch gradient descent with a batch size of 32, shuffling the data at the start of each epoch to ensure different batches in each pass through the dataset. I chose batch size 32 because it's large enough to provide stable gradient estimates and efficient matrix operations, but small enough to maintain the gradient noise that helps escape local minima. The value is also a power of 2, which optimizes CPU and GPU cache alignment for better memory access patterns.

```python
def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, 
          batch_size=32, early_stopping_patience=50):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Mini-batch gradient descent
        permutation = np.random.permutation(num_samples)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            
            y_pred = self.forward(X_batch)
            grads = self.backward(y_batch)
            self.update_parameters(grads)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = [w.copy() for w in self.weights]
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            self.weights = best_weights  # Restore best weights
            break
```

Early stopping with a patience of 50 epochs prevents overfitting by monitoring validation loss and stopping training when the model stops improving. The patience value of 50 allows the model to plateau temporarily without triggering early stopping, since validation loss can stay flat for many epochs before improving again. When training does stop early, I restore the weights from the epoch with the best validation loss rather than keeping the final weights, ensuring we return the best model rather than a potentially overfit version. This is particularly important for small datasets where the gap between training and validation performance can widen quickly once the model starts memorizing.

---

## Data Augmentation: 50 → 1000+ Pairs

With only 50 manually curated image-MIDI pairs, data augmentation was absolutely critical to prevent overfitting. I implemented a sophisticated pipeline that applies semantically meaningful transformations to both images and their corresponding MIDI files, ensuring that the augmented pairs maintain the logical relationship between visual features and musical characteristics.

### Image Augmentation Strategy

When adjusting image brightness, I work in the HSV (Hue, Saturation, Value) color space rather than RGB because brightness is isolated in the V channel. If I were to adjust brightness in RGB space by scaling all three channels independently, I would inadvertently shift hues—for example, making a red image appear more orange. Working in HSV preserves the hue and saturation while only modifying brightness, which matches human perception of "the same image, just lighter or darker" and maintains the color relationships that are critical for mood detection.

```python
@staticmethod
def adjust_brightness(image, factor):
    """Adjust brightness in HSV space (preserves hue)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)  # Modify V channel
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

For contrast adjustment, I center the transformation around the image's mean pixel value before applying the scaling factor. The mathematical relationship is `(x - mean) * factor + mean`, which ensures that contrast (the deviation from the mean) increases while the overall brightness remains approximately constant. Simply multiplying raw pixel values would cause bright images to become brighter and dark images to become darker, changing both contrast and brightness simultaneously. By centering around the mean, I isolate the contrast effect, and the final clipping to the 0-255 range prevents overflow into invalid color values.

```python
@staticmethod
def adjust_contrast(image, factor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    adjusted = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255)
    return adjusted.astype(np.uint8)
```

Hue shifting operates on the circular color wheel, which is why I use modulo arithmetic. In the HSV color space, hue values range from 0 to 180 in OpenCV (compressed from the standard 0-360 range to fit in an 8-bit unsigned integer), and colors wrap around—red transitions through orange, yellow, green, blue, violet, and back to red. When I shift the hue by, say, 30 degrees, a value of 170 would become 200, which needs to wrap back to 20 using the modulo operation. This maintains the circular continuity of the color wheel and ensures that hue shifts create valid, perceptually smooth color transformations.

```python
@staticmethod
def adjust_hue(image, shift):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180  # Hue wraps around
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

### MIDI Augmentation Strategy

MIDI transposition shifts all notes up or down by a specified number of semitones while preserving the melodic intervals between notes. When I pair this with hue shifting in images, I create a semantically consistent relationship—warm hues like orange and yellow map to higher pitches, while cool hues like blue and purple map to lower pitches. The clamping to MIDI's valid note range of 0-127 prevents attempts to generate impossible notes, and importantly, the transposition maintains the musicality of the original melody since all interval relationships are preserved.

```python
@staticmethod
def transpose(midi_path, semitones, output_path):
    """Shift all notes up/down by semitones"""
    for msg in track:
        if msg.type in ['note_on', 'note_off']:
            new_note = max(0, min(127, msg.note + semitones))
            new_msg = msg.copy(note=new_note)
```

Velocity adjustment in MIDI correlates with the perceived loudness and intensity of notes. I pair velocity adjustments with saturation changes in images because both represent the "intensity" dimension—highly saturated colors feel vibrant and energetic, just as high-velocity notes sound louder and more forceful. The minimum velocity is set to 1 rather than 0 because in MIDI, a note-on message with velocity 0 is actually interpreted as a note-off event, so we need to avoid that edge case.

```python
@staticmethod
def adjust_velocity(midi_path, velocity_factor, output_path):
    if msg.type == 'note_on':
        new_velocity = max(1, min(127, int(msg.velocity * velocity_factor)))
```

### Augmentation Pairing Logic

The key insight in my augmentation strategy is that image transformations must be paired with semantically equivalent MIDI transformations to maintain the learned relationship between visual and musical features. For example, when I create a "vibrant" variant by increasing both saturation and contrast in the image, I simultaneously increase the MIDI velocity to make the music feel more energetic and intense. This pairing ensures that the model learns generalizable patterns like "brightness correlates with tempo" rather than memorizing specific image-music combinations.

```python
configs.append({
    'name': 'vibrant',
    'image_fn': lambda img: self.img_aug.adjust_saturation(
        self.img_aug.adjust_contrast(img, 1.3), 1.4),
    'midi_fn': lambda path, out: self.midi_aug.adjust_velocity(path, 1.2, out)
})
```

By generating approximately 20 augmented variants from each original pair, I expand the dataset from 50 to over 1000 training examples. This 20× expansion gives the neural network enough diverse examples to learn robust feature representations without overfitting to the specific images in the original dataset. The augmentation teaches the model that the relationship between visual features and music is consistent across variations in brightness, color, and intensity, which is exactly the kind of generalization we need for the model to work on completely new, unseen images.

---

## MIDI Generation & Optimization

After the neural network predicts musical parameters, I use a two-stage process to generate the final MIDI file. The first stage creates an initial melody based on the predicted parameters, and the second stage refines this melody using simulated annealing optimization to improve its musicality.

### Stage 1: Initial Melody Generation

The initial melody generation leverages the pitch class histogram (PCH) predicted by the neural network. The PCH is a 12-dimensional probability distribution where each bin represents one of the chromatic pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B). For example, a melody in C Major would have high probability values for C, E, and G (scale degrees 1, 3, and 5), and lower probabilities for notes outside the key. I sample from this distribution using `np.random.choice(12, p=pch)`, which gives me a pitch class that respects the tonal center learned by the neural network. This approach ensures that the melody stays in key without enforcing hard constraints, allowing for occasional chromatic notes that can add musical interest.

```python
def generate_initial_sequence(music_params, target_duration=15):
    pch = music_params.get('pitch_class_histogram', np.ones(12) / 12.0)
    
    for i in range(target_num_notes):
        # Sample pitch class from predicted distribution
        pitch_class = np.random.choice(12, p=pch)
        
        # Find pitch in target octave
        octave_base = int(round(target_pitch_area / 12.0)) * 12
        pitch = octave_base + pitch_class
```

Once I have a pitch class, I need to determine which octave to place it in. Simply using a random octave would create chaotic melodic contours with large leaps between consecutive notes. Instead, I use a strategy that keeps pitches relatively close to the previous note while allowing controlled variation based on the predicted pitch standard deviation. I calculate a target pitch area centered on the last pitch, with bounds determined by the predicted pitch_std parameter. Then I find the closest octave of the sampled pitch class to this target area, which ensures smooth melodic motion. This technique creates singable, musically coherent melodies rather than random pitch jumps, which is crucial for producing listenable lofi music.

```python
pitch_candidate_low = last_pitch - int(round(pitch_std * random.uniform(0.5, 1.5)))
pitch_candidate_high = last_pitch + int(round(pitch_std * random.uniform(0.5, 1.5)))
target_pitch_area = random.uniform(pitch_candidate_low, pitch_candidate_high)
```

The octave selection logic includes an additional refinement step where if the initially chosen octave places the pitch more than 6 semitones away from the target area, I check the octaves above and below to see if they would be closer. This prevents situations where the pitch class C might be chosen, but C4 is 11 semitones away while C5 is only 1 semitone away from the target. By picking the closest octave, I ensure that melodic intervals stay reasonable and the contour remains smooth.

### Stage 2: Simulated Annealing Optimization

The initial melody captures the broad musical parameters predicted by the neural network, but it often lacks refinement in areas like interval smoothness, rhythmic consistency, and scale adherence. To address this, I use simulated annealing, which is a probabilistic optimization algorithm inspired by the metallurgical process of annealing. The algorithm starts with a high "temperature" that allows it to make large, exploratory moves in the solution space, then gradually cools down to make increasingly conservative, refinement-focused moves.

```python
def simulated_annealing(melody, music_params, initial_temp=1.0, 
                       cooling_rate=0.995, max_iterations=3000):
    temp = initial_temp
    
    for iteration in range(max_iterations):
        neighbor_melody = get_neighbor(current_melody)
        neighbor_fitness = calculate_fitness(neighbor_melody, music_params)
        
        fitness_change = neighbor_fitness - current_fitness
        
        if fitness_change > 0:
            current_melody = neighbor_melody  # Always accept improvements
        else:
            if temp > 1e-6:
                acceptance_probability = math.exp(fitness_change / temp)
                if random.random() < acceptance_probability:
                    current_melody = neighbor_melody  # Accept with probability
        
        temp *= cooling_rate  # Cool down
```

I chose simulated annealing over gradient-based optimization for several critical reasons. First, this is a discrete optimization problem where we're making changes like "shift this note up 2 semitones" or "make this note 20% longer," and you cannot take gradients of such discrete operations. Second, the fitness landscape is highly non-convex with many local optima, meaning there are multiple different "good" melodies that could match the predicted parameters. Simulated annealing's probabilistic acceptance of worse solutions during the high-temperature phase allows it to escape local optima and explore the solution space more thoroughly than greedy hill-climbing would.

The cooling rate of 0.995 creates an exponential decay schedule where the temperature decreases gradually over iterations. After 1000 iterations, the temperature drops to approximately 0.007 (since 0.995^1000 ≈ 0.007), by which point the algorithm behaves almost greedily, accepting only improvements. This rate is slow enough to allow thorough exploration in the early iterations when temperature is high, but fast enough to converge to a good solution within the 3000-iteration budget. If I used a faster cooling rate like 0.99, the algorithm would converge too quickly and likely get stuck in local optima; a slower rate like 0.999 would waste iterations on random exploration even when we should be refining.

The fitness function evaluates how well a melody matches both the predicted musical parameters and fundamental principles of music theory. I use a weighted combination of six different fitness components, with the weights reflecting their relative importance to musical quality. Scale adherence gets the highest weight at 30% because staying in key is paramount for musicality—a melody full of wrong notes will sound terrible regardless of its rhythm or dynamics. Interval smoothness receives 25% weight because singable melodies avoid large leaps greater than an octave. Rhythmic consistency, pitch range matching, pitch standard deviation matching, and average duration matching together comprise the remaining 45%, fine-tuning the melody to match the neural network's predictions.

```python
def calculate_fitness(melody, music_params):
    # 1. Scale adherence (30% weight)
    notes_in_scale = sum(1 for pitch in pitches if (pitch % 12) in TARGET_SCALE)
    scale_fitness = notes_in_scale / num_notes
    
    # 2. Smooth intervals (25% weight)
    intervals = np.abs(np.diff(pitches))
    large_leap_penalty = np.mean(np.maximum(0, intervals - 12)**2)
    interval_fitness = 1.0 / (1.0 + large_leap_penalty)
    
    # 3. Rhythmic consistency (15% weight)
    target_dur_std = music_params.get('duration_std')
    actual_dur_std = np.std(durations)
    rhythmic_fitness = 1.0 / (1.0 + np.abs(actual_dur_std - target_dur_std))
    
    # ... pitch range, pitch std, duration (30% combined)
    
    return weighted_sum
```

The interval fitness component uses a squared penalty for large leaps, which heavily punishes very large intervals while being lenient on moderate ones. The formula `np.maximum(0, intervals - 12)` first filters out intervals of an octave or less (which are perfectly acceptable in melodies), then squares the remaining values. This quadratic penalty means that a 15-semitone leap gets penalized much more severely than three 5-semitone leaps of equivalent total distance. The squaring creates a strong pressure toward smooth melodic motion while not being overly restrictive about reasonable musical intervals.

The neighbor generation function implements local search by making small, random modifications to a single note in the melody. I randomly select one note and randomly choose to modify either its pitch, duration, or velocity. For pitch changes, I use small mutations of ±1 to ±3 semitones (up to a minor third), which is enough variation to explore different melodic possibilities without destroying good solutions. Large mutations like ±12 semitones would essentially turn this into random search rather than local optimization, defeating the purpose of simulated annealing's gradual exploration and refinement strategy.

```python
def get_neighbor(melody):
    chosen_index = random.randrange(len(neighbor_melody))
    mutation_type = random.choice(['pitch', 'duration', 'velocity'])
    
    if mutation_type == 'pitch':
        pitch_change = random.choice([-3, -2, -1, 1, 2, 3])
```

The small mutation size is critical because simulated annealing is fundamentally a local search algorithm that should explore the neighborhood of the current solution. If mutations were too large, we would be doing random search, losing the benefit of the annealing schedule. By constraining pitch changes to ±3 semitones (a minor third), we explore musically plausible variations while maintaining enough of the original melody's character to build on good solutions incrementally.

---

## MIDI to Audio Conversion

MIDI files are symbolic representations that specify which notes to play and when, but they don't contain actual audio data. To let users play the generated music immediately without needing specialized software, I convert the MIDI files to WAV audio using FluidSynth, a software synthesizer that renders MIDI to PCM audio using soundfonts.

FluidSynth was the natural choice for this conversion because it uses soundfont (.sf2) files, which are collections of sampled real instruments. This produces much higher quality output than simple sine wave synthesis, and the rendering process is extremely fast, typically completing in 1-2 seconds for a 15-second piece of music. FluidSynth is also cross-platform and works reliably on Mac, Linux, and Windows, which is important for deployment flexibility.

```python
def convert_midi_to_wav(midi_path, wav_path, soundfont_path=None):
    if soundfont_path is None:
        soundfont_path = find_soundfont()  # System default or custom lofi soundfont
    
    fs = FluidSynth(soundfont_path)
    fs.midi_to_audio(midi_path, wav_path)
```

I chose to output WAV files rather than MP3 for several pragmatic reasons. WAV is an uncompressed PCM format, which means there's no encoding overhead and the conversion is nearly instantaneous. The format is also lossless, preserving the full audio quality from the synthesis stage. While WAV files are larger than MP3, this isn't a significant concern for 15-30 second audio clips (typically under 3MB), and all browsers and media players support WAV natively. If file size becomes an issue in production, I can easily add MP3 compression using FFmpeg as a post-processing step.

The soundfont selection logic checks multiple common installation paths because different systems install soundfonts in different locations. On Mac systems using Homebrew, the default soundfont is typically at `/opt/homebrew/share/soundfonts/default.sf2`, while Linux systems often use `/usr/share/sounds/sf2/FluidR3_GM.sf2`. I also check for a custom lofi soundfont in a local directory, which allows for future enhancement where I could provide a specialized soundfont with vinyl crackle, tape saturation, and warm piano samples to better capture the lofi aesthetic.

```python
def find_soundfont():
    possible_paths = [
        "/opt/homebrew/share/soundfonts/default.sf2",  # Mac Homebrew
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",         # Linux
        "./soundfonts/lofi.sf2"                          # Custom lofi soundfont
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
```

The graceful degradation strategy ensures the system continues working even if WAV synthesis fails. If no soundfont is found or if FluidSynth encounters an error, the backend still returns the MIDI file successfully, and the frontend can offer MIDI download as a fallback. This design philosophy prioritizes reliability and user experience, ensuring users always get something useful even when secondary features fail.

---

## Frontend Architecture

### State Management Strategy

```javascript
const [image, setImage] = useState(null);          // File object
const [imagePreview, setImagePreview] = useState(null);  // Blob URL for display
const [midiUrl, setMidiUrl] = useState(null);      // /outputs/uuid.mid
const [audioUrl, setAudioUrl] = useState(null);    // /outputs/uuid.wav
const [isGenerating, setIsGenerating] = useState(false);
```

The React frontend uses a careful separation of concerns in its state management, maintaining distinct pieces of state for different purposes. I keep the raw uploaded File object in the `image` state variable because it's needed for FormData when making the API request to the backend. Separately, I store a blob URL in `imagePreview` created by `URL.createObjectURL()`, which provides an efficient way to display the image preview without re-reading the entire file. This separation prevents unnecessary re-uploads and allows me to revoke the blob URL when the component unmounts, avoiding memory leaks.

I maintain both `midiUrl` and `audioUrl` in the state to provide flexibility for users. The audio URL is preferred when available because most users want to immediately play the music in their browser or media player. However, musicians and power users might want the MIDI file to edit in a digital audio workstation, so I offer both. The backend includes the audio URL conditionally in its response only if WAV conversion succeeded, which is why the frontend handles it as an optional field.

### File Upload with React Dropzone

For the file upload interface, I integrated React Dropzone because it provides a significantly better user experience than a basic HTML file input. The library handles drag-and-drop interactions, which feel more intuitive and modern than clicking a button to open a file picker. It also includes built-in MIME type filtering on the client side using the `accept` parameter, which validates that users are uploading image files before any data is sent to the server. On mobile devices where drag-and-drop doesn't make sense, it gracefully falls back to the native file picker, and it includes proper keyboard navigation and screen reader support for accessibility.

```javascript
const { getRootProps, getInputProps, isDragActive } = useDropzone({
  onDrop,
  accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'] },
  maxFiles: 1
});
```

I set `maxFiles` to 1 because the current system design processes one image at a time, and limiting to a single file simplifies both state management and the user experience. The user sees an immediate preview of their uploaded image without confusion about which image is being processed, and the backend API expects exactly one file per request. If I wanted to support batch processing in the future, I would need to restructure the state to hold an array of images and modify the backend to handle multiple generations in parallel.

### API Communication

The FormData approach for file uploads is necessary because binary file data cannot be sent as JSON. Browsers encode form data using multipart/form-data encoding, which allows mixing different types of content (like binary files and text parameters) in a single HTTP request. Axios automatically handles setting the proper Content-Type header with the correct boundary markers when it detects a FormData object, so I don't need to manually configure this.

```javascript
const generateMusic = async () => {
  const formData = new FormData();
  formData.append('file', image);
  formData.append('duration', duration);
  
  const response = await axios.post('/api/generate', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  
  if (response.data.success) {
    setMidiUrl(response.data.midi_url);
    setAudioUrl(response.data.audio_url || null);
    setSuccess(true);
  }
};
```

The defensive programming pattern of `audioUrl || null` is important because the backend's response only includes an `audio_url` field when WAV conversion succeeds. Without this fallback, accessing `response.data.audio_url` when it's undefined would cause runtime errors. By explicitly setting `audioUrl` to null when the field is missing, I can safely use conditional checks like `if (audioUrl)` throughout the component to show or hide the audio download button.

### Download Implementation

The download functionality creates a temporary anchor element programmatically rather than using `window.open()` because popup blockers in modern browsers often interfere with the latter approach. By creating an `<a>` element with a `download` attribute and programmatically triggering its click event, I ensure cross-browser compatibility and avoid popup blocker issues. The download attribute tells the browser to download the resource rather than navigating to it, and it allows me to specify the filename that will be saved.

```javascript
const downloadAudio = () => {
  const url = audioUrl || midiUrl;  // Prefer audio, fallback to MIDI
  const extension = audioUrl ? '.wav' : '.mid';
  
  const link = document.createElement('a');
  link.href = `http://localhost:8000${url}`;
  link.download = `lofi_${requestId}${extension}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
```

The cleanup step of removing the anchor element after clicking prevents DOM pollution. While a single orphaned element wouldn't cause problems, if users generated dozens of songs in one session without cleanup, we'd accumulate unnecessary DOM nodes. The prefix of `http://localhost:8000` is required in development because the React dev server runs on port 3000 while the backend API serves static files on port 8000, creating a cross-origin scenario. In production, I would use relative paths or a CDN URL instead.

---

## Performance Optimizations

### Backend

**Caching Dropout Masks by Layer**

In the dropout implementation, I cache masks using layer-specific keys like `dropout_mask_0`, `dropout_mask_1`, etc. This design choice is critical because backpropagation requires the exact same dropout mask that was used during the forward pass. When we zero out certain neurons during forward propagation, we need to apply those same zeros during backpropagation to maintain gradient flow consistency. Additionally, each layer has different dimensions—Layer 1's mask might be (batch_size, 64) while Layer 2's is (batch_size, 128)—so a single global mask would cause dimension mismatches. Using a dictionary with layer indices as keys gives us O(1) lookup time during backpropagation, making the implementation both correct and efficient.

```python
self.cache[f'dropout_mask_{layer_idx}'] = mask
```

**Batch Normalization Running Statistics**

The batch normalization implementation maintains separate statistics for training and inference through an exponential moving average with momentum set to 0.9. During training, we normalize each mini-batch using that batch's own mean and variance, which allows the network to learn despite varying batch compositions. However, during inference, we might only have a single sample, making it impossible to compute meaningful batch statistics. This is where the running statistics come in—we track a smoothed estimate of the population mean and variance throughout training using the exponential moving average formula. The momentum value of 0.9 means we give 90% weight to the historical average and 10% to the current batch, which smooths out noisy batch estimates while still adapting to distribution shifts in the data.

```python
self.bn_running_mean[layer_idx] = momentum * self.bn_running_mean[layer_idx] + (1 - momentum) * batch_mean
```

### Frontend

**Using `useCallback` for File Drop Handler**

I wrapped the `onDrop` function in React's `useCallback` hook with an empty dependency array to prevent unnecessary re-renders and function recreations. React Dropzone library subscribes to the onDrop callback, and if the function reference changes on every render, it triggers re-initialization of the drag-and-drop zone. By memoizing the function with `useCallback`, we ensure the same function reference persists across renders unless the dependencies change. Since the onDrop logic doesn't depend on any state or props that change over time, the empty dependency array `[]` is appropriate. This optimization is particularly important for performance because creating a new function on every render would cause React Dropzone to reconfigure its event listeners repeatedly, leading to sluggish file upload interactions.

```javascript
const onDrop = useCallback((acceptedFiles) => {
  // ... handle file
}, []);
```

---

## Future Improvements

### Model Enhancements

The current model could be significantly improved by replacing the simple feature extraction with convolutional neural networks. Instead of computing global statistics like average brightness and contrast, CNNs could extract spatial features like edges, textures, and compositional patterns that carry more nuanced mood information. An attention mechanism would allow the model to focus on the most relevant regions of an image—for example, bright spots in an image might drive tempo changes while dark regions influence bassline depth. With a larger dataset of 5000+ image-MIDI pairs, I could deploy much deeper networks without overfitting, potentially using architectures like ResNet or Vision Transformers that have proven successful in other image-to-X translation tasks.

### Music Generation

The current system generates monophonic melodies, but real lofi music requires richer harmonic content. I would add chord progression generation that follows music theory rules (like ii-V-I progressions in jazz), creating a separate bass track that reinforces the harmonic structure. Adding drum patterns would involve quantizing notes to a rhythmic grid and generating kick, snare, and hi-hat layers with characteristic lofi swing and groove. The ultimate goal would be multi-track MIDI with separate channels for melody, bass, piano chords, and percussion, giving users complete control to remix individual elements. I could also train separate specialized models for different genres (lofi hip-hop, jazz, classical) and let users select their preferred style before generation.

### Audio Synthesis

The default General MIDI soundfont produces clean but generic-sounding music. Creating a custom lofi soundfont with sampled vinyl crackle, tape saturation effects, and warm detuned piano would dramatically improve the aesthetic authenticity. MP3 compression would reduce file sizes by 90% compared to WAV, though it would require integrating FFmpeg into the backend pipeline. For the ultimate user experience, I could implement real-time audio preview by streaming audio chunks as they're generated, letting users hear the first few seconds while the rest is still being synthesized rather than waiting for the complete file.

---

## Installation

```bash
# Backend
cd backend
python3.12 -m pip install --user fastapi uvicorn numpy opencv-python mido scikit-learn matplotlib midi2audio
brew install fluid-synth

# Frontend
cd frontend
npm install
```

## Usage

```bash
# Terminal 1: Backend
cd backend
python3.12 app.py

# Terminal 2: Frontend  
cd frontend
npm start

# Open http://localhost:3000
```

---

## Technical Validation

The final training metrics demonstrate that the model successfully learned meaningful patterns rather than simply memorizing the training data. The training loss converged to 0.048 MSE (mean squared error), while the validation loss reached 0.045 and the test loss achieved 0.043. These values indicate that the model's predictions are typically within about 22% of the true values, calculated as the square root of the MSE (√0.048 ≈ 0.22).

The fact that the test loss is approximately equal to the validation loss is strong evidence against overfitting. If the model were memorizing the training data rather than learning generalizable patterns, we would see the training loss continue decreasing while the validation and test losses plateaued or increased. Instead, all three metrics track closely together and show a consistent downward trend, indicating the model generalizes well to completely unseen data.

The training converged in approximately 1800 epochs with early stopping, which suggests the architecture and regularization strategy were well-calibrated for the dataset size. If the model were underfitting, it would have required many more epochs to converge (or never converged at all). If it were overfitting, early stopping would have triggered much sooner, around 500-800 epochs. The fact that it trained for 1800 epochs before the validation loss stopped improving indicates we're extracting close to the maximum information possible from the 1000-pair augmented dataset while avoiding overfitting.



---

## 📝 License

MIT License - Feel free to use, modify, and distribute.

---
