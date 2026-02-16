# 🎵 Lofi Generator

**Transform images into lofi beats with AI**

An AI-powered web application that generates unique lofi music from images. Upload any image, and our neural network analyzes its visual features (colors, contrast, patterns) to create a matching lofi beat as a MIDI file.

---

## ✨ Features

- 🖼️ **Image Upload**: Drag-and-drop or browse to upload images
- 🤖 **AI Generation**: Deep neural network predicts musical parameters from image features
- 🎵 **MIDI Output**: Download professional MIDI files compatible with any DAW
- 🔥 **Optimization**: Uses simulated annealing to refine melodies for better musicality
- 🎨 **Modern UI**: Beautiful, responsive interface with gradient backgrounds
- 📱 **Mobile-Friendly**: Works seamlessly on desktop and mobile devices
- 🚀 **Fast**: Generates music in seconds

---

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **Neural Network**: Custom 4-layer deep learning model with ReLU activation, batch normalization, and dropout
- **Image Processing**: OpenCV for feature extraction (brightness, contrast, RGB values, edge density)
- **MIDI Processing**: Mido library for MIDI file generation and manipulation
- **Optimization**: Simulated annealing algorithm to optimize melodies
- **API**: RESTful FastAPI endpoints for generation and downloads

### Frontend (React)
- **Modern UI**: React with hooks and modern JavaScript
- **File Upload**: React Dropzone for drag-and-drop
- **Styling**: Custom CSS with gradients and animations
- **HTTP Client**: Axios for API communication

### Data Pipeline
1. **Image Features** (6 features): brightness, contrast, mean R/G/B, edge density
2. **MIDI Features** (20 features): tempo, pitch statistics, pitch class histogram, velocity, rhythm, duration
3. **Training**: 1000+ augmented image-MIDI pairs
4. **Prediction**: Scaled neural network output → music parameters
5. **Generation**: Initial melody → simulated annealing optimization → MIDI file

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker (optional, for containerized deployment)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd lofi-generator

# Build and run with docker-compose
docker-compose up --build

# Access the app at http://localhost:3000
```

### Option 2: Manual Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

Backend will run on `http://localhost:8000`

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend will run on `http://localhost:3000`

---

## 📚 Training the Model

Before using the app, you need to train the model with your image-MIDI pairs.

### Step 1: Prepare Your Data

Create a list of training pairs (50+ pairs recommended):

```python
training_pairs = [
    {'image_path': 'path/to/image1.jpg', 'midi_path': 'path/to/midi1.mid'},
    {'image_path': 'path/to/image2.jpg', 'midi_path': 'path/to/midi2.mid'},
    # ... add all your pairs
]
```

### Step 2: Run Training

```python
from backend.training_pipeline import run_complete_training_pipeline

# Train the model (will automatically augment to 1000+ pairs)
model, scaler_x, scaler_y, history = run_complete_training_pipeline(
    original_pairs=training_pairs,
    augment=True,
    augmentation_target=1000,
    epochs=2000,
    batch_size=32,
    early_stopping_patience=100
)
```

This will:
1. ✅ Augment your 50 pairs to 1000+ pairs
2. ✅ Extract features from all pairs
3. ✅ Train the neural network
4. ✅ Save the model to `./models/`
5. ✅ Generate training plots

### Step 3: Model is Ready!

The trained model will be automatically loaded by the backend when you start the server.

---

## 📖 Usage Guide

### Web Interface

1. **Upload Image**: Click or drag-and-drop an image (JPG, PNG, GIF, WebP)
2. **Set Duration**: Choose the duration (10-60 seconds)
3. **Generate**: Click "Generate Lofi Beat"
4. **Download**: Download the generated MIDI file

### API Endpoints

#### Generate Music
```bash
POST /api/generate
Content-Type: multipart/form-data

Parameters:
- file: Image file
- duration: Duration in seconds (optional, default: 15)
- sa_iterations: Optimization iterations (optional, default: 3000)

Response:
{
  "success": true,
  "request_id": "uuid",
  "midi_url": "/outputs/uuid.mid",
  "filename": "uuid.mid",
  "duration": 15,
  "timestamp": "2024-02-15T10:30:00"
}
```

#### Download MIDI
```bash
GET /api/download/{request_id}

Response: MIDI file download
```

#### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-02-15T10:30:00"
}
```

---

## 🎨 Data Augmentation

The system includes powerful data augmentation to expand your dataset:

### Image Augmentations
- Brightness adjustments (0.7x - 1.3x)
- Contrast variations (0.8x - 1.4x)
- Saturation changes (0.6x - 1.6x)
- Hue shifts (-30° to +30°)
- Noise addition
- Gaussian blur
- Rotations (-15° to +15°)
- Horizontal flips
- Center crops

### MIDI Augmentations
- Transposition (-3 to +3 semitones)
- Tempo changes (0.9x - 1.2x)
- Velocity adjustments (0.8x - 1.2x)

### Running Augmentation Separately

```python
from backend.data_augmentation import run_augmentation

augmented_pairs = run_augmentation(
    original_pairs,
    output_dir='./augmented_data',
    target_total=1000
)
```

---

## 🧠 Model Architecture

### Neural Network
- **Input Layer**: 6 neurons (image features)
- **Hidden Layer 1**: 64 neurons + ReLU + Batch Norm + Dropout(0.3)
- **Hidden Layer 2**: 128 neurons + ReLU + Batch Norm + Dropout(0.3)
- **Hidden Layer 3**: 128 neurons + ReLU + Batch Norm + Dropout(0.3)
- **Hidden Layer 4**: 64 neurons + ReLU + Batch Norm
- **Output Layer**: 20 neurons (music features) + Sigmoid

### Training Features
- **Optimizer**: Gradient Descent with momentum
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: Dropout (0.3), Batch Normalization
- **Early Stopping**: Patience of 100 epochs
- **Learning Rate**: 0.001
- **Batch Size**: 32

### Music Generation
1. **Prediction**: Neural network outputs 20 music parameters
2. **Initial Generation**: Create melody based on parameters
3. **Optimization**: Simulated annealing refines melody for:
   - Scale adherence (C major)
   - Smooth intervals
   - Rhythmic consistency
   - Target pitch range and variation
   - Duration consistency
4. **MIDI Export**: Convert optimized melody to MIDI format

---

## 📂 Project Structure

```
lofi-generator/
├── backend/
│   ├── app.py                      # FastAPI server
│   ├── improved_model.py           # Neural network implementation
│   ├── training_pipeline.py        # Complete training pipeline
│   ├── data_augmentation.py        # Data augmentation utilities
│   ├── midi_generation.py          # MIDI generation & optimization
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Backend Docker config
│   ├── models/                     # Trained models (created after training)
│   ├── uploads/                    # Temporary uploaded images
│   └── outputs/                    # Generated MIDI files
├── frontend/
│   ├── src/
│   │   ├── App.js                  # Main React component
│   │   ├── App.css                 # Styles
│   │   ├── index.js                # React entry point
│   │   └── index.css               # Global styles
│   ├── public/
│   │   └── index.html              # HTML template
│   ├── package.json                # Node dependencies
│   ├── Dockerfile                  # Frontend Docker config
│   └── nginx.conf                  # Nginx configuration
├── docker-compose.yml              # Docker Compose config
└── README.md                       # This file
```

---

## 🎯 Performance & Results

### Training Metrics
- **Training Loss**: Typically converges to ~0.01-0.03
- **Validation Loss**: ~0.02-0.04
- **Training Time**: ~30-60 minutes on CPU (1000 pairs, 2000 epochs)
- **Model Size**: ~500KB

### Generation Speed
- **Feature Extraction**: <1 second
- **Neural Network Prediction**: <0.1 seconds
- **MIDI Optimization**: 2-5 seconds (3000 iterations)
- **Total Generation Time**: ~3-6 seconds

---

## 🔧 Configuration

### Backend Configuration

Edit `backend/app.py` to adjust:
- Upload directory
- Output directory
- Model path
- API host/port

### Model Hyperparameters

Edit `backend/improved_model.py` to adjust:
- Hidden layer sizes
- Learning rate
- Dropout rate
- Activation functions

### Optimization Parameters

Edit `backend/midi_generation.py` to adjust:
- Simulated annealing iterations
- Initial temperature
- Cooling rate
- Fitness weights

---

## 🚀 Deployment

### Railway.app

1. Push to GitHub
2. Connect Railway to your repo
3. Add two services: backend and frontend
4. Set environment variables
5. Deploy!

### Vercel (Frontend) + Railway (Backend)

**Frontend (Vercel):**
```bash
cd frontend
vercel deploy
```

**Backend (Railway):**
```bash
cd backend
railway up
```

### Docker Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🤝 Contributing

Contributions welcome! Here are some ways to improve the project:

### Ideas for Enhancement
- [ ] Add more music scales (minor, pentatonic, blues)
- [ ] Support for longer compositions (>60 seconds)
- [ ] Real-time audio preview (convert MIDI to audio)
- [ ] Style selection (chill, upbeat, melancholic)
- [ ] Batch processing (multiple images)
- [ ] User accounts and saved generations
- [ ] Social sharing features
- [ ] Mobile app (React Native)

### Code Improvements
- [ ] Add unit tests
- [ ] Implement caching for predictions
- [ ] Add more image preprocessing techniques
- [ ] Experiment with different neural architectures (CNN, Transformer)
- [ ] Add progress indicators for long generations
- [ ] Implement queue system for multiple requests

---

## 📝 License

MIT License - feel free to use this project for personal or commercial purposes.

---

## 🙏 Acknowledgments

- **Mido**: MIDI library for Python
- **FastAPI**: Modern web framework
- **React**: Frontend framework
- **OpenCV**: Image processing
- **Scikit-learn**: Machine learning utilities

---

## 📞 Support

Having issues? Here are common solutions:

### Model not loading
```bash
# Ensure you've trained the model first
python -c "from backend.training_pipeline import run_complete_training_pipeline; ..."
```

### Port already in use
```bash
# Backend
lsof -ti:8000 | xargs kill -9

# Frontend
lsof -ti:3000 | xargs kill -9
```

### Dependencies issues
```bash
# Backend
pip install --upgrade -r requirements.txt

# Frontend
rm -rf node_modules package-lock.json
npm install
```

---

## 🎉 Get Started Now!

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd lofi-generator

# 2. Train the model with your data
python backend/train_model.py

# 3. Start the app
docker-compose up

# 4. Open http://localhost:3000 and create music! 🎵
```

---

Made with ❤️ and AI | [GitHub](https://github.com/yourusername/lofi-generator)
