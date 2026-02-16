"""
Improved Neural Network for Image-to-Music Generation
- Deeper architecture (4 hidden layers)
- ReLU activation
- Dropout for regularization
- Batch normalization
- Early stopping
"""

import numpy as np
import warnings

class ImprovedNeuralNetwork:
    def __init__(self, input_size=6, hidden_sizes=[64, 128, 128, 64], output_size=20, 
                 learning_rate=0.001, dropout_rate=0.3):
        """
        Initialize improved neural network with multiple hidden layers
        
        Args:
            input_size: Number of input features (6 image features)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output features (20 MIDI features)
            learning_rate: Learning rate for optimization
            dropout_rate: Dropout probability for regularization
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.training_mode = True
        
        # Initialize weights and biases using He initialization (better for ReLU)
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # He initialization: sqrt(2/n) where n is the number of input units
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Batch normalization parameters
        self.bn_gamma = [np.ones((1, size)) for size in hidden_sizes]
        self.bn_beta = [np.zeros((1, size)) for size in hidden_sizes]
        self.bn_running_mean = [np.zeros((1, size)) for size in hidden_sizes]
        self.bn_running_var = [np.ones((1, size)) for size in hidden_sizes]
        
        # Cache for backpropagation
        self.cache = {}
        
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation for output layer"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def batch_norm(self, x, gamma, beta, running_mean, running_var, layer_idx, momentum=0.9):
        """
        Batch normalization
        
        Args:
            x: Input to normalize
            gamma: Scale parameter
            beta: Shift parameter
            running_mean: Running mean for inference
            running_var: Running variance for inference
            layer_idx: Index of the layer
            momentum: Momentum for running statistics
        """
        if self.training_mode:
            # Training mode: use batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            self.bn_running_mean[layer_idx] = momentum * self.bn_running_mean[layer_idx] + (1 - momentum) * batch_mean
            self.bn_running_var[layer_idx] = momentum * self.bn_running_var[layer_idx] + (1 - momentum) * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + 1e-8)
            
            # Cache for backprop
            self.cache[f'bn_mean_{layer_idx}'] = batch_mean
            self.cache[f'bn_var_{layer_idx}'] = batch_var
            self.cache[f'bn_x_norm_{layer_idx}'] = x_norm
            self.cache[f'bn_x_{layer_idx}'] = x
        else:
            # Inference mode: use running statistics
            x_norm = (x - running_mean) / np.sqrt(running_var + 1e-8)
        
        # Scale and shift
        out = gamma * x_norm + beta
        return out
    
    def dropout(self, x, rate, layer_idx):
        """
        Dropout for regularization
        
        Args:
            x: Input
            rate: Dropout probability
            layer_idx: Index of the layer (for caching separate masks)
        """
        if self.training_mode and rate > 0:
            mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
            self.cache[f'dropout_mask_{layer_idx}'] = mask
            return x * mask
        return x
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input features (batch_size, input_size)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        self.cache['X'] = X
        current_input = X
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.cache[f'z_{i}'] = z
            
            # Batch normalization
            z_bn = self.batch_norm(z, self.bn_gamma[i], self.bn_beta[i], 
                                   self.bn_running_mean[i], self.bn_running_var[i], i)
            
            # ReLU activation
            a = self.relu(z_bn)
            self.cache[f'a_{i}'] = a
            
            # Dropout (only in training)
            if i < len(self.weights) - 2:  # Don't apply dropout to last hidden layer
                a = self.dropout(a, self.dropout_rate, i)
            
            current_input = a
        
        # Output layer (sigmoid activation for bounded output)
        z_out = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid(z_out)
        
        self.cache['z_out'] = z_out
        self.cache['output'] = output
        
        return output
    
    def backward(self, y_true):
        """
        Backward pass to compute gradients
        
        Args:
            y_true: True target values
        """
        if y_true.shape != self.cache['output'].shape:
            y_true = y_true.reshape(self.cache['output'].shape)
        
        m = self.cache['X'].shape[0]
        grads = {}
        
        # Output layer gradient
        dz_out = self.cache['output'] - y_true
        grads[f'dW_{len(self.weights)-1}'] = np.dot(self.cache[f'a_{len(self.weights)-2}'].T, dz_out) / m
        grads[f'db_{len(self.weights)-1}'] = np.sum(dz_out, axis=0, keepdims=True) / m
        
        # Backpropagate through hidden layers
        da = np.dot(dz_out, self.weights[-1].T)
        
        for i in range(len(self.weights) - 2, -1, -1):
            # Dropout gradient (use layer-specific mask)
            if i < len(self.weights) - 2 and self.training_mode:
                mask_key = f'dropout_mask_{i}'
                if mask_key in self.cache:
                    da = da * self.cache[mask_key]
            
            # ReLU gradient
            dz = da * self.relu_derivative(self.cache[f'a_{i}'])
            
            # Batch norm gradient (simplified)
            dz_bn = dz  # Simplified, full BN backprop is more complex
            
            # Weight and bias gradients
            if i > 0:
                grads[f'dW_{i}'] = np.dot(self.cache[f'a_{i-1}'].T, dz_bn) / m
            else:
                grads[f'dW_{i}'] = np.dot(self.cache['X'].T, dz_bn) / m
            
            grads[f'db_{i}'] = np.sum(dz_bn, axis=0, keepdims=True) / m
            
            # Batch norm parameter gradients
            grads[f'dgamma_{i}'] = np.sum(dz * self.cache[f'bn_x_norm_{i}'], axis=0, keepdims=True) / m
            grads[f'dbeta_{i}'] = np.sum(dz, axis=0, keepdims=True) / m
            
            # Propagate to previous layer
            if i > 0:
                da = np.dot(dz_bn, self.weights[i].T)
        
        return grads
    
    def update_parameters(self, grads):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads[f'dW_{i}']
            self.biases[i] -= self.learning_rate * grads[f'db_{i}']
            
            # Update batch norm parameters
            if i < len(self.bn_gamma):
                self.bn_gamma[i] -= self.learning_rate * grads[f'dgamma_{i}']
                self.bn_beta[i] -= self.learning_rate * grads[f'dbeta_{i}']
    
    def mean_squared_error(self, y_true, y_pred):
        """Calculate MSE loss"""
        if y_true.shape != y_pred.shape:
            y_true = y_true.reshape(y_pred.shape)
        return np.mean((y_true - y_pred) ** 2)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, 
              batch_size=32, early_stopping_patience=50, verbose=True):
        """
        Train the network with early stopping
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Maximum number of epochs
            batch_size: Mini-batch size
            early_stopping_patience: Stop if no improvement for N epochs
            verbose: Print training progress
        """
        self.training_mode = True
        history = {'train_loss': [], 'val_loss': []}
        
        num_samples = X_train.shape[0]
        batch_size = min(batch_size, num_samples) if batch_size > 0 else num_samples
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        if verbose:
            print(f"Training started: {epochs} epochs, LR={self.learning_rate}, "
                  f"BatchSize={batch_size}, Dropout={self.dropout_rate}")
            print(f"Architecture: {self.input_size} -> {' -> '.join(map(str, self.hidden_sizes))} -> {self.output_size}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle training data
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                loss = self.mean_squared_error(y_batch, y_pred)
                epoch_loss += loss * X_batch.shape[0]
                
                # Backward pass
                grads = self.backward(y_batch)
                
                # Update parameters
                self.update_parameters(grads)
            
            # Calculate average training loss
            avg_train_loss = epoch_loss / num_samples
            history['train_loss'].append(avg_train_loss)
            
            # Validation loss
            if X_val is not None and y_val is not None:
                self.training_mode = False
                val_pred = self.forward(X_val)
                val_loss = self.mean_squared_error(y_val, val_pred)
                history['val_loss'].append(val_loss)
                self.training_mode = True
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    best_weights = [w.copy() for w in self.weights]
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    # Restore best weights
                    if best_weights is not None:
                        self.weights = best_weights
                    break
                
                # Print progress
                if verbose and ((epoch + 1) % 100 == 0 or epoch == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                # No validation set
                if verbose and ((epoch + 1) % 100 == 0 or epoch == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
        
        if verbose:
            print("Training finished.")
        
        return history
    
    def predict(self, X):
        """Make predictions (inference mode)"""
        self.training_mode = False
        predictions = self.forward(X)
        self.training_mode = True
        return predictions
    
    def save_model(self, filepath):
        """Save model weights and parameters"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'bn_gamma': self.bn_gamma,
            'bn_beta': self.bn_beta,
            'bn_running_mean': self.bn_running_mean,
            'bn_running_var': self.bn_running_var,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate
        }
        np.save(filepath, model_data)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights and parameters"""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.bn_gamma = model_data['bn_gamma']
        self.bn_beta = model_data['bn_beta']
        self.bn_running_mean = model_data['bn_running_mean']
        self.bn_running_var = model_data['bn_running_var']
        self.input_size = model_data['input_size']
        self.hidden_sizes = model_data['hidden_sizes']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        self.dropout_rate = model_data['dropout_rate']
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    print("ImprovedNeuralNetwork class defined successfully!")
