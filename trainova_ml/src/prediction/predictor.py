"""
Trainova ML - Workout Predictor
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from ..models.ml_models import (
    RandomForestWorkoutModel,
    NeuralNetworkWorkoutModel,
    LSTMWorkoutModel
)
from ..utils.model_utils import (
    calculate_one_rep_max,
    calculate_weight_for_reps,
    extract_workout_features,
    generate_feedback_message
)


class MLWorkoutPredictor:
    """
    Machine Learning based workout predictor that uses actual ML models
    to predict workout weights and incorporate user feedback.
    """
    
    def __init__(self, model_type: str = 'neural_network'):
        """
        Initialize the ML workout predictor.
        
        Args:
            model_type: Type of ML model to use ('random_forest', 'neural_network', 'lstm')
        """
        self.model_type = model_type
        self.feedback_history = []
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestWorkoutModel(model_name='trainova_rf_model')
        elif model_type == 'neural_network':
            self.model = NeuralNetworkWorkoutModel(model_name='trainova_nn_model')
        elif model_type == 'lstm':
            self.model = LSTMWorkoutModel(model_name='trainova_lstm_model')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Try to load model if it exists
        self.model.load_model()
        
        # Initialize feature columns and exercises
        self.feature_columns = None
        self.trained_exercises = set()
    
    def fit_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ML model on workout data.
        
        Args:
            training_data: DataFrame containing workout data
            
        Returns:
            Dictionary with training metrics
        """
        if training_data.empty:
            return {"success": False, "message": "No training data provided"}
        
        # Start timer to measure actual training time
        start_time = datetime.now()
        
        # Process data for each exercise separately
        all_exercises = training_data['exercise'].unique()
        all_metrics = {}
        
        # Store all trained exercises
        self.trained_exercises = set(all_exercises)
        
        # Create a combined dataset to ensure consistent features
        combined_X = pd.DataFrame()
        combined_y = pd.Series(dtype=float)
        
        for exercise in all_exercises:
            # Filter data for this exercise
            exercise_data = training_data[training_data['exercise'] == exercise].copy()
            
            if len(exercise_data) < 3:
                print(f"Not enough data for {exercise}, skipping...")
                continue
            
            # Extract features for model training
            X = extract_workout_features(exercise_data)
            y = exercise_data['weight']
            
            # Append to combined dataset
            if not X.empty:
                combined_X = pd.concat([combined_X, X])
                combined_y = pd.concat([combined_y, y])
        
        # Save feature columns for prediction
        self.feature_columns = combined_X.columns.tolist()
        
        # Ensure no NaN values exist in the data
        combined_X = combined_X.fillna(0)
        combined_y = combined_y.fillna(combined_y.mean())
        
        # Train model on combined dataset
        print(f"Training model with {len(combined_X)} records across {len(all_exercises)} exercises...")
        metrics = self.model.fit(combined_X, combined_y)
        
        # Calculate total training time
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"Model training completed in {training_duration:.2f} seconds for {len(all_exercises)} exercises")
        
        return {
            "success": True,
            "message": f"Model trained successfully with {len(training_data)} records",
            "training_time": training_duration,
            "metrics": metrics
        }
    
    def predict_workout(self, user_data: dict, debug: bool = False) -> dict:
        """
        Make a prediction for the next workout weight
        
        Args:
            user_data: Dictionary with user workout data
            debug: Whether to include debug information in the output
        
        Returns:
            Dictionary with prediction results
        """
        exercise = user_data.get('exercise', '')
        last_weight = float(user_data.get('last_weight', 0))
        
        # Create a DataFrame with user data
        user_df = pd.DataFrame([user_data])
        
        try:
            # Check if the model has been trained
            if not hasattr(self, 'feature_columns') or self.feature_columns is None:
                raise ValueError("Model has not been trained yet - no feature columns available")
                
            # Check if the exercise was in our training data
            if not hasattr(self, 'trained_exercises') or exercise not in self.trained_exercises:
                # For new exercises, we'll fall back to a simpler prediction method
                return {
                    'predicted_weight': last_weight * 1.02,  # 2% increase
                    'confidence': 0.3,
                    'suggested_reps': 5,
                    'message': f"New exercise detected. Try increasing to {last_weight * 1.02:.1f}kg for your next workout.",
                    'method': 'new_exercise_fallback'
                }
                
            # Extract features for this workout
            features = extract_workout_features(user_df)
            
            # Create a DataFrame with all expected columns initialized to 0
            prediction_features = pd.DataFrame(0, index=[0], columns=self.feature_columns)
            
            # Fill in values for columns that exist in the features DataFrame
            for col in features.columns:
                if col in prediction_features.columns:
                    prediction_features[col] = features[col].values
            
            # Handle exercise one-hot encoding specifically
            for col in prediction_features.columns:
                if col.startswith('exercise_'):
                    prediction_features[col] = 0
            
            # Set the current exercise column to 1
            exercise_col = f'exercise_{exercise}'
            if exercise_col in prediction_features.columns:
                prediction_features[exercise_col] = 1
            
            # Fill any remaining NaN values
            features = prediction_features.fillna(0)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Sanity check: prediction shouldn't be too far from last weight
            # Prevent unrealistic low predictions
            if prediction < last_weight * 0.9:  # If prediction is more than 10% lower
                # Use a more conservative approach: 98-100% of last weight
                prediction = last_weight * (0.98 + np.random.random() * 0.02)
                confidence = 0.30  # Lower confidence for this adjusted prediction
            # Prevent unrealistic high predictions
            elif prediction > last_weight * 1.1:  # If prediction is more than 10% higher
                # Use a more conservative approach: 2-5% increase
                prediction = last_weight * (1.02 + np.random.random() * 0.03)
                confidence = 0.35  # Lower confidence for this adjusted prediction
            else:
                # For predictions within reasonable range, use higher confidence
                confidence = 0.65
            
            # Determine if we should increase, decrease, or maintain the weight
            if prediction > last_weight:
                message = f"Try increasing to {prediction:.1f}kg for your next workout."
            elif prediction < last_weight:
                message = f"Consider decreasing to {prediction:.1f}kg for your next workout."
            else:
                message = f"Maintain current weight of {prediction:.1f}kg for your next workout."
            
            # Calculate suggested reps based on prediction vs last weight
            suggested_reps = 5  # Default suggestion
            
            result = {
                'predicted_weight': prediction,
                'confidence': confidence,
                'suggested_reps': suggested_reps,
                'message': message,
                'method': 'model_prediction'
            }
            
            # Include debug information if requested
            if debug:
                result['debug_info'] = {
                    'features': features.to_dict(),
                    'model_params': self.model.get_params()
                }
            
            return result
        except Exception as e:
            print(f"Prediction error: {e}")
            
            # Fallback prediction based on last weight
            # Simple 1-2% increase as fallback
            fallback_prediction = last_weight * 1.017
            
            return {
                'predicted_weight': fallback_prediction,
                'confidence': 0.40,
                'suggested_reps': 5,
                'message': f"Try increasing to {fallback_prediction:.1f}kg for your next workout.",
                'method': 'fallback',
                'last_weight': last_weight,
                'error': str(e)
            }
    
    def record_feedback(self, exercise: str, predicted_weight: float, actual_weight: float, 
                      success: bool, reps: int = None, rir: int = None) -> Dict[str, Any]:
        """
        Record user feedback about a prediction to improve future predictions.
        
        Args:
            exercise: Exercise name
            predicted_weight: Weight that was predicted
            actual_weight: Weight that was actually used
            success: Whether the workout was successful
            reps: Number of reps completed
            rir: Reps in reserve
            
        Returns:
            Dictionary with feedback results
        """
        # Calculate score based on difference between predicted and actual
        weight_diff = actual_weight - predicted_weight
        relative_diff = weight_diff / predicted_weight if predicted_weight > 0 else 0
        score = max(min(relative_diff, 1.0), -1.0)
        
        # Adjust score based on RIR
        if rir is not None:
            # Higher RIR means the weight was easier than expected
            if rir > 2:
                score += 0.2
            elif rir > 0:
                score += 0.1
            else:
                score -= 0.1
        
        # Adjust score based on success
        if not success:
            score -= 0.3
        
        # Record feedback
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'exercise': exercise,
            'predicted_weight': predicted_weight,
            'actual_weight': actual_weight,
            'success': success,
            'reps': reps,
            'rir': rir,
            'score': score
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Generate feedback message
        message = generate_feedback_message(score)
        
        return {
            'feedback_recorded': True,
            'score': round(score, 3),
            'message': message
        }
    
    def reset_model(self, reset_type: str = 'all') -> Dict[str, Any]:
        """
        Reset the model data.
        
        Args:
            reset_type: Type of reset to perform ('all', 'model', 'feedback')
            
        Returns:
            Dictionary with reset results
        """
        result = {"success": False, "message": ""}
        
        try:
            if reset_type in ['all', 'model']:
                # Get model directory
                model_dir = self.model.model_dir
                
                # Delete model files
                for filename in os.listdir(model_dir):
                    if filename.startswith(self.model.model_name):
                        file_path = os.path.join(model_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                
                # Create a new model instance
                if self.model_type == 'random_forest':
                    self.model = RandomForestWorkoutModel(model_name='trainova_rf_model')
                elif self.model_type == 'neural_network':
                    self.model = NeuralNetworkWorkoutModel(model_name='trainova_nn_model')
                elif self.model_type == 'lstm':
                    self.model = LSTMWorkoutModel(model_name='trainova_lstm_model')
                
                result["message"] += "Model reset successful. "
            
            if reset_type in ['all', 'feedback']:
                # Clear feedback history
                self.feedback_history = []
                result["message"] += "Feedback history reset successful. "
            
            result["success"] = True
            
        except Exception as e:
            result["message"] = f"Error during reset: {e}"
        
        return result
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the model performance on test data.
        
        Args:
            test_data: DataFrame containing test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if test_data.empty:
            return {"success": False, "message": "No test data provided"}
        
        # Process data for each exercise separately
        all_exercises = test_data['exercise'].unique()
        all_metrics = {}
        combined_actual = []
        combined_predicted = []
        
        for exercise in all_exercises:
            # Filter data for this exercise
            exercise_data = test_data[test_data['exercise'] == exercise].copy()
            
            if len(exercise_data) < 3:
                print(f"Not enough test data for {exercise}, skipping evaluation...")
                continue
            
            # Extract features for evaluation
            X = extract_workout_features(exercise_data)
            y_actual = exercise_data['weight'].values
            
            # Create a DataFrame with all expected columns initialized to 0
            if not hasattr(self, 'feature_columns') or self.feature_columns is None:
                print(f"Model has not been trained yet, skipping evaluation for {exercise}")
                continue
                
            evaluation_features = pd.DataFrame(0, index=range(len(X)), columns=self.feature_columns)
            
            # Fill in values for columns that exist in the features DataFrame
            for col in X.columns:
                if col in evaluation_features.columns:
                    evaluation_features[col] = X[col].values
            
            # Handle exercise one-hot encoding specifically
            for col in evaluation_features.columns:
                if col.startswith('exercise_'):
                    evaluation_features[col] = 0
            
            # Set the current exercise column to 1
            exercise_col = f'exercise_{exercise}'
            if exercise_col in evaluation_features.columns:
                evaluation_features[exercise_col] = 1
            
            # Fill any remaining NaN values
            X_eval = evaluation_features.fillna(0)
            
            # Make predictions
            y_pred = self.model.predict(X_eval)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
            mae = np.mean(np.abs(y_actual - y_pred))
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
            
            # Store metrics for this exercise
            all_metrics[exercise] = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'count': len(exercise_data)
            }
            
            # Add to combined metrics
            combined_actual.extend(y_actual)
            combined_predicted.extend(y_pred)
        
        # Calculate overall metrics
        if combined_actual:
            combined_actual = np.array(combined_actual)
            combined_predicted = np.array(combined_predicted)
            
            overall_rmse = np.sqrt(np.mean((combined_actual - combined_predicted) ** 2))
            overall_mae = np.mean(np.abs(combined_actual - combined_predicted))
            overall_mape = np.mean(np.abs((combined_actual - combined_predicted) / combined_actual)) * 100
            
            overall_metrics = {
                'rmse': overall_rmse,
                'mae': overall_mae,
                'mape': overall_mape,
                'count': len(combined_actual)
            }
        else:
            overall_metrics = {
                'rmse': None,
                'mae': None,
                'mape': None,
                'count': 0
            }
        
        return {
            'success': True,
            'message': f"Model evaluated on {len(test_data)} records across {len(all_exercises)} exercises",
            'overall_metrics': overall_metrics,
            'exercise_metrics': all_metrics
        }
    
    def switch_model_type(self, new_model_type: str) -> Dict[str, Any]:
        """
        Switch to a different type of model.
        
        Args:
            new_model_type: Type of ML model to use ('random_forest', 'neural_network', 'lstm')
            
        Returns:
            Dictionary with switch results
        """
        result = {"success": False, "message": ""}
        
        try:
            if new_model_type not in ['random_forest', 'neural_network', 'lstm']:
                result["message"] = f"Unknown model type: {new_model_type}"
                return result
            
            if new_model_type == self.model_type:
                result["message"] = f"Already using {new_model_type} model"
                result["success"] = True
                return result
            
            # Update model type
            self.model_type = new_model_type
            
            # Create a new model instance
            if new_model_type == 'random_forest':
                self.model = RandomForestWorkoutModel(model_name='trainova_rf_model')
            elif new_model_type == 'neural_network':
                self.model = NeuralNetworkWorkoutModel(model_name='trainova_nn_model')
            elif new_model_type == 'lstm':
                self.model = LSTMWorkoutModel(model_name='trainova_lstm_model')
            
            # Try to load model if it exists
            self.model.load_model()
            
            result["message"] = f"Successfully switched to {new_model_type} model"
            result["success"] = True
            
        except Exception as e:
            result["message"] = f"Error during model switch: {e}"
        
        return result