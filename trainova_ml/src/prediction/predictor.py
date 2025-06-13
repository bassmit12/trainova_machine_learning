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
                # Load existing training data and train the model if not already trained
                training_data = self._load_all_training_data()
                if not training_data.empty:
                    self.fit_model(training_data)
                else:
                    return {
                        "predicted_weight": last_weight + 2.5,
                        "confidence": 0.3,
                        "message": "No training data available. Using simple progression.",
                        "suggested_reps": [8],
                        "suggested_sets": 3
                    }
            
            # Get the last 10 sets of exercise history for this exercise
            exercise_history = self._get_exercise_history(exercise, limit=10)
            
            if debug:
                print(f"\nDebug - Processing {len(exercise_history)} recent sets for {exercise}")
                if len(exercise_history) > 0:
                    print(f"Latest set: {exercise_history[-1]['weight']}kg x {exercise_history[-1]['reps']} reps")
                    print(f"Date range: {exercise_history[0]['date']} to {exercise_history[-1]['date']}")
            
            # If we have historical data, use it for prediction
            if len(exercise_history) >= 3:
                # Extract features from the last 10 sets
                features_df = self._extract_features_from_history(exercise_history, exercise)
                
                if debug:
                    print(f"Extracted features shape: {features_df.shape}")
                    print(f"Feature columns: {list(features_df.columns)}")
                
                # Make prediction using the model
                prediction = self.model.predict(features_df)
                predicted_weight = float(prediction[0]) if len(prediction) > 0 else last_weight + 2.5
                
                # Calculate confidence based on data quality and model performance
                confidence = self._calculate_confidence(exercise_history, predicted_weight)
                
                # Generate suggested reps based on progression
                suggested_reps = self._generate_rep_suggestions(exercise_history, predicted_weight)
                
                # Ensure minimum progression
                if predicted_weight <= last_weight:
                    predicted_weight = last_weight + 2.5
                
                # Round to nearest 2.5kg increment
                predicted_weight = round(predicted_weight / 2.5) * 2.5
                
                message = f"Prediction based on {len(exercise_history)} recent sets"
                
                if debug:
                    print(f"Final prediction: {predicted_weight}kg with {confidence:.2f} confidence")
                
            else:
                # Fallback for insufficient data
                predicted_weight = last_weight + 2.5
                confidence = 0.4
                suggested_reps = [8]
                message = "Limited data available. Using conservative progression."
                
                if debug:
                    print(f"Insufficient data ({len(exercise_history)} sets). Using fallback prediction.")
            
            return {
                "predicted_weight": predicted_weight,
                "confidence": confidence,
                "message": message,
                "suggested_reps": suggested_reps,
                "suggested_sets": 3,
                "analysis": {
                    "data_points": len(exercise_history),
                    "last_weight": last_weight,
                    "progression": predicted_weight - last_weight
                } if debug else None
            }

        except Exception as e:
            if debug:
                print(f"Error in prediction: {str(e)}")
            
            return {
                "predicted_weight": last_weight + 2.5,
                "confidence": 0.2,
                "message": f"Prediction error: {str(e)}. Using fallback.",
                "suggested_reps": [8],
                "suggested_sets": 3
            }
    
    def _get_exercise_history(self, exercise: str, limit: int = None) -> List[Dict]:
        """
        Get ALL workout sets for a specific exercise (no limit for optimal predictions)
        
        Args:
            exercise: Exercise name
            limit: Ignored - we use all available data for optimal predictions
            
        Returns:
            List of ALL workout sets for the exercise
        """
        try:
            # Load all training data
            all_data = self._load_all_training_data()
            
            if all_data.empty:
                return []
            
            # Filter for the specific exercise
            exercise_data = all_data[all_data['exercise'] == exercise]
            
            if exercise_data.empty:
                return []
            
            # Sort by date and return ALL records (no limit)
            exercise_data = exercise_data.sort_values('date' if 'date' in exercise_data.columns else exercise_data.index)
            
            # Convert ALL data to list of dictionaries
            return exercise_data.to_dict('records')
            
        except Exception as e:
            print(f"Error getting exercise history: {e}")
            return []
    
    def _extract_features_from_history(self, history: List[Dict], exercise: str) -> pd.DataFrame:
        """
        Extract features from exercise history for model prediction
        
        Args:
            history: List of workout records
            exercise: Exercise name
            
        Returns:
            DataFrame with extracted features
        """
        if not history:
            return pd.DataFrame()
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(history)
        
        # Ensure required columns exist
        required_cols = ['weight', 'reps']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Calculate features
        features = {
            'last_weight': df['weight'].iloc[-1],
            'last_reps': df['reps'].iloc[-1],
            'avg_weight': df['weight'].mean(),
            'max_weight': df['weight'].max(),
            'weight_progression': (df['weight'].iloc[-1] - df['weight'].iloc[0]) if len(df) > 1 else 0,
            'avg_reps': df['reps'].mean(),
            'total_volume': (df['weight'] * df['reps']).sum(),
            'consistency_score': 1.0 / (1.0 + df['weight'].std()) if len(df) > 1 else 1.0,
            'session_count': len(df),
            'exercise_encoded': hash(exercise) % 1000  # Simple exercise encoding
        }
        
        # Add time-based features if date is available
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                features['days_since_last'] = (pd.Timestamp.now() - df['date'].iloc[-1]).days
                features['training_frequency'] = len(df) / max(1, (df['date'].iloc[-1] - df['date'].iloc[0]).days)
            except:
                features['days_since_last'] = 7  # Default
                features['training_frequency'] = 0.5  # Default
        else:
            features['days_since_last'] = 7
            features['training_frequency'] = 0.5
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure all feature columns are present (pad with zeros if needed)
        if hasattr(self, 'feature_columns'):
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Reorder columns to match training data
            features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
        
        return features_df
    
    def _calculate_confidence(self, history: List[Dict], predicted_weight: float) -> float:
        """
        Calculate confidence score based on data quality and prediction
        
        Args:
            history: Exercise history
            predicted_weight: Predicted weight
            
        Returns:
            Confidence score between 0 and 1
        """
        if not history:
            return 0.2
        
        base_confidence = 0.4
        
        # Data quantity factor
        data_factor = min(len(history) / 10, 0.3)  # Max 0.3 bonus for 10+ data points
        
        # Consistency factor
        weights = [h['weight'] for h in history]
        consistency_factor = 1.0 / (1.0 + np.std(weights)) * 0.2 if len(weights) > 1 else 0.1
        
        # Recency factor (if we have dates)
        recency_factor = 0.1  # Default
        
        total_confidence = base_confidence + data_factor + consistency_factor + recency_factor
        return min(total_confidence, 1.0)
    
    def _generate_rep_suggestions(self, history: List[Dict], predicted_weight: float) -> List[int]:
        """
        Generate rep suggestions based on history and predicted weight
        
        Args:
            history: Exercise history
            predicted_weight: Predicted weight
            
        Returns:
            List of suggested reps
        """
        if not history:
            return [8]
        
        # Analyze recent rep patterns
        recent_reps = [h.get('reps', 8) for h in history[-3:]]  # Last 3 sets
        avg_reps = sum(recent_reps) / len(recent_reps)
        
        # Base suggestion on average, with some variation
        if avg_reps >= 10:
            suggested_reps = [8, 6, 5]  # Suggest fewer reps for strength
        elif avg_reps <= 5:
            suggested_reps = [6, 8, 10]  # Suggest more reps for volume
        else:
            suggested_reps = [int(avg_reps), int(avg_reps) - 1, int(avg_reps) + 1]
        
        # Ensure reps are in reasonable range
        suggested_reps = [max(3, min(15, rep)) for rep in suggested_reps]
        
        return suggested_reps[:3]  # Return top 3 suggestions
    
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