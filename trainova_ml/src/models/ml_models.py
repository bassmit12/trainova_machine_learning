"""
Trainova ML - Machine Learning Models
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

# Replace TensorFlow imports with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..utils.model_utils import calculate_training_time


class BaseWorkoutModel:
    """Base class for all workout prediction models."""
    
    def __init__(self, model_name: str = 'base_model'):
        """
        Initialize the base workout model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    "data", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"{self.model_name}.joblib")
        self.scaler_X_path = os.path.join(self.model_dir, f"{self.model_name}_scaler_X.joblib")
        self.scaler_y_path = os.path.join(self.model_dir, f"{self.model_name}_scaler_y.joblib")
        self.training_history = []
        self.metrics = {}
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, 
                      fit_scalers: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for model training or prediction.
        
        Args:
            X: Features DataFrame
            y: Target Series (optional for prediction)
            fit_scalers: Whether to fit the scalers to the data
            
        Returns:
            Preprocessed X and y as numpy arrays
        """
        if fit_scalers:
            X_scaled = self.scaler_X.fit_transform(X)
            if y is not None:
                y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
            else:
                y_scaled = None
            
            # Save the scalers
            joblib.dump(self.scaler_X, self.scaler_X_path)
            joblib.dump(self.scaler_y, self.scaler_y_path)
        else:
            # Load scalers if they exist
            if os.path.exists(self.scaler_X_path) and os.path.exists(self.scaler_y_path):
                try:
                    self.scaler_X = joblib.load(self.scaler_X_path)
                    self.scaler_y = joblib.load(self.scaler_y_path)
                except Exception as e:
                    print(f"Error loading scalers: {e}")
            
            X_scaled = self.scaler_X.transform(X)
            if y is not None:
                y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()
            else:
                y_scaled = None
        
        return X_scaled, y_scaled
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the model on workout data.
        
        Args:
            X: Features DataFrame
            y: Target Series
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        # This should be implemented by the derived class
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weight predictions for workouts.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Numpy array of predictions
        """
        # This should be implemented by the derived class
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_model(self) -> None:
        """Save the model to disk."""
        # This should be implemented by the derived class
        raise NotImplementedError("Subclasses must implement this method")
    
    def load_model(self) -> bool:
        """
        Load the model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        # This should be implemented by the derived class
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with evaluation metrics
        """
        # This should be implemented by the derived class
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Update the model with new training data.
        
        Args:
            X: Features DataFrame
            y: Target Series
            **kwargs: Additional update parameters
            
        Returns:
            Dictionary with update metrics
        """
        # By default, we'll just retrain the model
        return self.fit(X, y, **kwargs)


class RandomForestWorkoutModel(BaseWorkoutModel):
    """Random Forest based workout prediction model."""
    
    def __init__(self, model_name: str = 'random_forest_model', n_estimators: int = 200, 
               max_depth: int = 15, random_state: int = 42):
        """
        Initialize the Random Forest workout model.
        
        Args:
            model_name: Name of the model
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random state for reproducibility
        """
        super().__init__(model_name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the Random Forest model on workout data.
        
        Args:
            X: Features DataFrame
            y: Target Series
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        # Start training timer
        start_time = datetime.now()
        
        # Preprocess data
        X_train, y_train = self.preprocess_data(X, y, fit_scalers=True)
        
        # Split data for validation if enough samples
        if len(X_train) > 10:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state
            )
            self.model.fit(X_train_split, y_train_split)
            
            # Evaluate on validation set
            y_val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            self.metrics = {
                'val_mse': val_mse,
                'val_r2': val_r2
            }
        else:
            # Train on all data if not enough samples
            self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Calculate training time
        training_time = calculate_training_time(start_time)
        
        # Save model
        self.save_model()
        
        # Record training history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(X),
            'train_mse': train_mse,
            'train_r2': train_r2,
            'training_time': training_time,
            **self.metrics
        }
        self.training_history.append(history_entry)
        
        return history_entry
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weight predictions using the Random Forest model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Numpy array of predictions
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess data
        X_scaled, _ = self.preprocess_data(X)
        
        # Make predictions
        y_scaled_pred = self.model.predict(X_scaled)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def save_model(self) -> None:
        """Save the Random Forest model to disk."""
        # Create a dictionary with all model components
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'metrics': self.metrics,
            'training_history': self.training_history,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }
        }
        
        # Save the model data
        joblib.dump(model_data, self.model_path)
    
    def load_model(self) -> bool:
        """
        Load the Random Forest model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            return False
        
        try:
            model_data = joblib.load(self.model_path)
            
            self.model = model_data['model']
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.metrics = model_data['metrics']
            self.training_history = model_data['training_history']
            
            hyperparameters = model_data.get('hyperparameters', {})
            self.n_estimators = hyperparameters.get('n_estimators', self.n_estimators)
            self.max_depth = hyperparameters.get('max_depth', self.max_depth)
            self.random_state = hyperparameters.get('random_state', self.random_state)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the Random Forest model performance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess data
        X_scaled, y_scaled = self.preprocess_data(X, y)
        
        # Make predictions
        y_scaled_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_scaled, y_scaled_pred)
        r2 = r2_score(y_scaled, y_scaled_pred)
        
        # Get feature importances
        feature_importances = {
            feature: importance
            for feature, importance in zip(X.columns, self.model.feature_importances_)
        }
        
        # Update metrics
        self.metrics = {
            'mse': mse,
            'r2': r2,
            'feature_importances': feature_importances
        }
        
        return self.metrics


# PyTorch Neural Network model
class PyTorchNN(nn.Module):
    """PyTorch Neural Network model for workout prediction."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout_rate: float = 0.2):
        """
        Initialize the PyTorch neural network.
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
        """
        super(PyTorchNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(nn.BatchNorm1d(hidden_layers[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], 1))
    
    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralNetworkWorkoutModel(BaseWorkoutModel):
    """Neural Network based workout prediction model using PyTorch."""
    
    def __init__(self, model_name: str = 'neural_network_model', 
               input_dim: int = 10, 
               hidden_layers: List[int] = [64, 32],
               dropout_rate: float = 0.2,
               learning_rate: float = 0.001,
               batch_size: int = 32,
               epochs: int = 100,
               patience: int = 10,
               random_state: int = 42):
        """
        Initialize the Neural Network workout model.
        
        Args:
            model_name: Name of the model
            input_dim: Input dimension (number of features)
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
            patience: Patience for early stopping
            random_state: Random state for reproducibility
        """
        super().__init__(model_name)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Override model paths for PyTorch models
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     "data", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"{self.model_name}.pt")
        self.scaler_X_path = os.path.join(self.model_dir, f"{self.model_name}_scaler_X.joblib")
        self.scaler_y_path = os.path.join(self.model_dir, f"{self.model_name}_scaler_y.joblib")
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def _build_model(self) -> PyTorchNN:
        """
        Build the neural network architecture.
        
        Returns:
            PyTorch neural network model
        """
        model = PyTorchNN(self.input_dim, self.hidden_layers, self.dropout_rate)
        model.to(self.device)
        return model
    
    def _train_epoch(self, model, dataloader, criterion, optimizer, is_train=True):
        """
        Train or evaluate for one epoch.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with batches
            criterion: Loss function
            optimizer: Optimizer
            is_train: Whether to train or evaluate
            
        Returns:
            Average loss for the epoch
        """
        epoch_loss = 0.0
        
        if is_train:
            model.train()
        else:
            model.eval()
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero the parameter gradients
            if is_train:
                optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(is_train):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                if is_train:
                    loss.backward()
                    optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss
        epoch_loss /= len(dataloader.dataset)
        
        return epoch_loss
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the Neural Network model on workout data.
        
        Args:
            X: Features DataFrame
            y: Target Series
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        # Start training timer
        start_time = datetime.now()
        
        # Update input dimension based on actual data
        self.input_dim = X.shape[1]
        
        # Preprocess data
        X_train, y_train = self.preprocess_data(X, y, fit_scalers=True)
        
        # Split data for validation if enough samples
        validation_data = None
        if len(X_train) > 10:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_split)
            y_train_tensor = torch.FloatTensor(y_train_split.reshape(-1, 1))
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
            
            # Create DataLoaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(self.batch_size, len(train_dataset)), 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=min(self.batch_size, len(val_dataset))
            )
            
            use_validation = True
        else:
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
            
            # Create DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(self.batch_size, len(train_dataset)), 
                shuffle=True
            )
            
            use_validation = False
        
        # Build the model
        self.model = self._build_model()
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.epochs):
            # Train on training set
            train_loss = self._train_epoch(self.model, train_loader, criterion, optimizer, is_train=True)
            history['train_loss'].append(train_loss)
            
            # Evaluate on validation set if available
            if use_validation:
                with torch.no_grad():
                    val_loss = self._train_epoch(self.model, val_loader, criterion, optimizer, is_train=False)
                history['val_loss'].append(val_loss)
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), self.model_path)
                else:
                    patience_counter += 1
                
                print(f'Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
                
                if patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                print(f'Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss:.4f}')
                # Save model at each epoch if no validation set
                torch.save(self.model.state_dict(), self.model_path)
        
        # Load best model if using validation
        if use_validation:
            self.model.load_state_dict(torch.load(self.model_path))
        
        # Calculate training time
        training_time = calculate_training_time(start_time)
        
        # Evaluate on training set
        self.model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
            y_train_pred = self.model(X_train_tensor).cpu().numpy().flatten()
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = np.mean(np.abs(y_train - y_train_pred))
        
        # Evaluate on validation set if available
        val_mse = None
        val_mae = None
        if use_validation:
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
                y_val_pred = self.model(X_val_tensor).cpu().numpy().flatten()
            
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_mae = np.mean(np.abs(y_val - y_val_pred))
        
        # Update metrics
        self.metrics = {
            'train_loss': train_loss,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'val_loss': best_val_loss if use_validation else None,
            'val_mse': val_mse,
            'val_mae': val_mae
        }
        
        # Save model
        self.save_model()
        
        # Record training history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(X),
            'epochs_trained': epoch + 1,
            'final_train_loss': train_loss,
            'final_train_mse': train_mse,
            'final_train_mae': train_mae,
            'final_val_loss': best_val_loss if use_validation else None,
            'final_val_mse': val_mse,
            'final_val_mae': val_mae,
            'training_time': training_time
        }
        self.training_history.append(history_entry)
        
        return history_entry
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weight predictions using the Neural Network model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Numpy array of predictions
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess data
        X_scaled, _ = self.preprocess_data(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_scaled_pred = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred).flatten()
        
        return y_pred
    
    def save_model(self) -> None:
        """Save the Neural Network model to disk."""
        # Save PyTorch model state dict
        torch.save(self.model.state_dict(), self.model_path)
        
        # Save scalers
        joblib.dump(self.scaler_X, self.scaler_X_path)
        joblib.dump(self.scaler_y, self.scaler_y_path)
        
        # Save metadata
        metadata = {
            'metrics': self.metrics,
            'training_history': self.training_history,
            'hyperparameters': {
                'input_dim': self.input_dim,
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience,
                'random_state': self.random_state
            }
        }
        joblib.dump(metadata, os.path.join(self.model_dir, f"{self.model_name}_metadata.joblib"))
    
    def load_model(self) -> bool:
        """
        Load the Neural Network model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            return False
        
        try:
            # Load metadata first to get hyperparameters
            metadata_path = os.path.join(self.model_dir, f"{self.model_name}_metadata.joblib")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.metrics = metadata.get('metrics', {})
                self.training_history = metadata.get('training_history', [])
                
                hyperparameters = metadata.get('hyperparameters', {})
                self.input_dim = hyperparameters.get('input_dim', self.input_dim)
                self.hidden_layers = hyperparameters.get('hidden_layers', self.hidden_layers)
                self.dropout_rate = hyperparameters.get('dropout_rate', self.dropout_rate)
                self.learning_rate = hyperparameters.get('learning_rate', self.learning_rate)
                self.batch_size = hyperparameters.get('batch_size', self.batch_size)
                self.epochs = hyperparameters.get('epochs', self.epochs)
                self.patience = hyperparameters.get('patience', self.patience)
                self.random_state = hyperparameters.get('random_state', self.random_state)
            
            # Create and load model
            self.model = self._build_model()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            # Load scalers
            if os.path.exists(self.scaler_X_path) and os.path.exists(self.scaler_y_path):
                self.scaler_X = joblib.load(self.scaler_X_path)
                self.scaler_y = joblib.load(self.scaler_y_path)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the Neural Network model performance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess data
        X_scaled, y_scaled = self.preprocess_data(X, y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled.reshape(-1, 1)).to(self.device)
        
        # Evaluate model
        self.model.eval()
        with torch.no_grad():
            y_scaled_pred = self.model(X_tensor).cpu().numpy().flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_scaled, y_scaled_pred)
        mae = np.mean(np.abs(y_scaled - y_scaled_pred))
        r2 = r2_score(y_scaled, y_scaled_pred)
        
        # Update metrics
        self.metrics = {
            'loss': mse,
            'mae': mae,
            'mse': mse,
            'r2': r2
        }
        
        return self.metrics


# PyTorch LSTM model
class PyTorchLSTM(nn.Module):
    """PyTorch LSTM model for time series workout prediction."""
    
    def __init__(self, input_size: int, hidden_size: List[int], dropout_rate: float = 0.2):
        """
        Initialize the PyTorch LSTM model.
        
        Args:
            input_size: Number of features per time step
            hidden_size: List of hidden units for each LSTM layer
            dropout_rate: Dropout rate for regularization
        """
        super(PyTorchLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size[0],
                batch_first=True,
                dropout=0 if len(hidden_size) == 1 else dropout_rate
            )
        )
        
        # Additional LSTM layers
        for i in range(1, len(hidden_size)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_size[i-1],
                    hidden_size=hidden_size[i],
                    batch_first=True,
                    dropout=0 if i == len(hidden_size)-1 else dropout_rate
                )
            )
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_size)):
            self.bn_layers.append(nn.BatchNorm1d(hidden_size[i]))
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(nn.Linear(hidden_size[-1], 16))
        self.dense_layers.append(nn.BatchNorm1d(16))
        self.dense_layers.append(nn.ReLU())
        self.dense_layers.append(nn.Dropout(dropout_rate))
        self.dense_layers.append(nn.Linear(16, 1))
    
    def forward(self, x):
        """Forward pass through the network."""
        batch_size = x.size(0)
        
        # Process through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:
                # Apply batch norm for intermediate layers only on the last time step
                x_last = x[:, -1, :]
                x_last = self.bn_layers[i](x_last)
                x = torch.cat([x[:, :-1, :], x_last.unsqueeze(1)], dim=1)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Apply batch norm for the last LSTM layer
        x = self.bn_layers[-1](x)
        
        # Process through dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        return x


class LSTMWorkoutModel(BaseWorkoutModel):
    """LSTM Neural Network for time series workout prediction using PyTorch."""
    
    def __init__(self, model_name: str = 'lstm_model', 
               sequence_length: int = 5,
               feature_dim: int = 10,
               lstm_units: List[int] = [64, 32],
               dense_units: List[int] = [16],
               dropout_rate: float = 0.2,
               learning_rate: float = 0.001,
               batch_size: int = 32,
               epochs: int = 100,
               patience: int = 10,
               random_state: int = 42):
        """
        Initialize the LSTM workout model.
        
        Args:
            model_name: Name of the model
            sequence_length: Length of input sequences
            feature_dim: Number of features per time step
            lstm_units: List of units in each LSTM layer
            dense_units: List of units in each Dense layer after LSTM
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
            patience: Patience for early stopping
            random_state: Random state for reproducibility
        """
        super().__init__(model_name)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Override model paths for PyTorch models
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     "data", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"{self.model_name}.pt")
        self.scaler_X_path = os.path.join(self.model_dir, f"{self.model_name}_scaler_X.joblib")
        self.scaler_y_path = os.path.join(self.model_dir, f"{self.model_name}_scaler_y.joblib")
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def _build_model(self) -> PyTorchLSTM:
        """
        Build the LSTM network architecture.
        
        Returns:
            PyTorch LSTM model
        """
        model = PyTorchLSTM(self.feature_dim, self.lstm_units, self.dropout_rate)
        model.to(self.device)
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create input sequences for LSTM model.
        
        Args:
            X: Feature array
            y: Target array (optional)
            
        Returns:
            Sequences for X and y (if provided)
        """
        n_samples = X.shape[0]
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(n_samples - self.sequence_length + 1):
            X_seq.append(X[i:i+self.sequence_length])
            if y is not None:
                y_seq.append(y[i+self.sequence_length-1])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def _train_epoch(self, model, dataloader, criterion, optimizer, is_train=True):
        """
        Train or evaluate for one epoch.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with batches
            criterion: Loss function
            optimizer: Optimizer
            is_train: Whether to train or evaluate
            
        Returns:
            Average loss for the epoch
        """
        epoch_loss = 0.0
        
        if is_train:
            model.train()
        else:
            model.eval()
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero the parameter gradients
            if is_train:
                optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(is_train):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                if is_train:
                    loss.backward()
                    optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss
        epoch_loss /= len(dataloader.dataset)
        
        return epoch_loss
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model on workout data.
        
        Args:
            X: Features DataFrame
            y: Target Series
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        # Start training timer
        start_time = datetime.now()
        
        # Update feature dimension based on actual data
        self.feature_dim = X.shape[1]
        
        # Preprocess data
        X_scaled, y_scaled = self.preprocess_data(X, y, fit_scalers=True)
        
        # Create sequences for LSTM
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Check if we have enough sequences
        if len(X_seq) < 2:
            raise ValueError(f"Not enough samples to create sequences. Need at least {self.sequence_length + 1} samples.")
        
        # Split data for validation if enough samples
        use_validation = False
        if len(X_seq) > 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=self.random_state
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
            
            # Create DataLoaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(self.batch_size, len(train_dataset)), 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=min(self.batch_size, len(val_dataset))
            )
            
            use_validation = True
        else:
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_seq)
            y_train_tensor = torch.FloatTensor(y_seq.reshape(-1, 1))
            
            # Create DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(self.batch_size, len(train_dataset)), 
                shuffle=True
            )
            
            # Set validation data for evaluation
            X_train, y_train = X_seq, y_seq
        
        # Build the model
        self.model = self._build_model()
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.epochs):
            # Train on training set
            train_loss = self._train_epoch(self.model, train_loader, criterion, optimizer, is_train=True)
            history['train_loss'].append(train_loss)
            
            # Evaluate on validation set if available
            if use_validation:
                with torch.no_grad():
                    val_loss = self._train_epoch(self.model, val_loader, criterion, optimizer, is_train=False)
                history['val_loss'].append(val_loss)
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), self.model_path)
                else:
                    patience_counter += 1
                
                print(f'Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
                
                if patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                print(f'Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss:.4f}')
                # Save model at each epoch if no validation set
                torch.save(self.model.state_dict(), self.model_path)
        
        # Load best model if using validation
        if use_validation:
            self.model.load_state_dict(torch.load(self.model_path))
        
        # Calculate training time
        training_time = calculate_training_time(start_time)
        
        # Evaluate on training set
        self.model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
            y_train_pred = self.model(X_train_tensor).cpu().numpy().flatten()
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = np.mean(np.abs(y_train - y_train_pred))
        
        # Evaluate on validation set if available
        val_mse = None
        val_mae = None
        if use_validation:
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
                y_val_pred = self.model(X_val_tensor).cpu().numpy().flatten()
            
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_mae = np.mean(np.abs(y_val - y_val_pred))
        
        # Update metrics
        self.metrics = {
            'train_loss': train_loss,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'val_loss': best_val_loss if use_validation else None,
            'val_mse': val_mse,
            'val_mae': val_mae
        }
        
        # Save model
        self.save_model()
        
        # Record training history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(X),
            'sequences': len(X_seq),
            'epochs_trained': epoch + 1,
            'final_train_loss': train_loss,
            'final_train_mse': train_mse,
            'final_train_mae': train_mae,
            'final_val_loss': best_val_loss if use_validation else None,
            'final_val_mse': val_mse,
            'final_val_mae': val_mae,
            'training_time': training_time
        }
        self.training_history.append(history_entry)
        
        return history_entry
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weight predictions using the LSTM model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Numpy array of predictions
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess data
        X_scaled, _ = self.preprocess_data(X)
        
        # Check if we have enough data for a sequence
        if len(X_scaled) < self.sequence_length:
            # Pad with zeros if not enough data
            padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_scaled_pred = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred).flatten()
        
        return y_pred[-1:]  # Return only the last prediction
    
    def save_model(self) -> None:
        """Save the LSTM model to disk."""
        # Save PyTorch model state dict
        torch.save(self.model.state_dict(), self.model_path)
        
        # Save scalers
        joblib.dump(self.scaler_X, self.scaler_X_path)
        joblib.dump(self.scaler_y, self.scaler_y_path)
        
        # Save metadata
        metadata = {
            'metrics': self.metrics,
            'training_history': self.training_history,
            'hyperparameters': {
                'sequence_length': self.sequence_length,
                'feature_dim': self.feature_dim,
                'lstm_units': self.lstm_units,
                'dense_units': self.dense_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience,
                'random_state': self.random_state
            }
        }
        joblib.dump(metadata, os.path.join(self.model_dir, f"{self.model_name}_metadata.joblib"))
    
    def load_model(self) -> bool:
        """
        Load the LSTM model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            return False
        
        try:
            # Load metadata first to get hyperparameters
            metadata_path = os.path.join(self.model_dir, f"{self.model_name}_metadata.joblib")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.metrics = metadata.get('metrics', {})
                self.training_history = metadata.get('training_history', [])
                
                hyperparameters = metadata.get('hyperparameters', {})
                self.sequence_length = hyperparameters.get('sequence_length', self.sequence_length)
                self.feature_dim = hyperparameters.get('feature_dim', self.feature_dim)
                self.lstm_units = hyperparameters.get('lstm_units', self.lstm_units)
                self.dense_units = hyperparameters.get('dense_units', self.dense_units)
                self.dropout_rate = hyperparameters.get('dropout_rate', self.dropout_rate)
                self.learning_rate = hyperparameters.get('learning_rate', self.learning_rate)
                self.batch_size = hyperparameters.get('batch_size', self.batch_size)
                self.epochs = hyperparameters.get('epochs', self.epochs)
                self.patience = hyperparameters.get('patience', self.patience)
                self.random_state = hyperparameters.get('random_state', self.random_state)
            
            # Create and load model
            self.model = self._build_model()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            # Load scalers
            if os.path.exists(self.scaler_X_path) and os.path.exists(self.scaler_y_path):
                self.scaler_X = joblib.load(self.scaler_X_path)
                self.scaler_y = joblib.load(self.scaler_y_path)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the LSTM model performance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess data
        X_scaled, y_scaled = self.preprocess_data(X, y)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Evaluate model
        if len(X_seq) > 0:
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            y_tensor = torch.FloatTensor(y_seq.reshape(-1, 1)).to(self.device)
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_tensor).cpu().numpy().flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_seq, y_pred)
            mae = np.mean(np.abs(y_seq - y_pred))
            r2 = r2_score(y_seq, y_pred)
            
            # Update metrics
            self.metrics = {
                'loss': mse,
                'mae': mae,
                'mse': mse,
                'r2': r2
            }
        else:
            print("Not enough data to create sequences for evaluation.")
            self.metrics = {}
        
        return self.metrics