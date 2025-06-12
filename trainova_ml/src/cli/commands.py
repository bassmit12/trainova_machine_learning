"""
Trainova ML - CLI Commands
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..prediction.predictor import MLWorkoutPredictor
from ..data.data_collection import DataCollector

class CommandHandler:
    """
    Handles the execution of CLI commands for the ML-based Trainova system.
    """
    
    def __init__(self):
        """Initialize the command handler with data collector and predictor."""
        self.data_collector = DataCollector()
        self.predictor = MLWorkoutPredictor()
    
    def handle_pretrain(self, args: argparse.Namespace) -> None:
        """
        Handle the pretrain command to pretrain the model with existing data.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Pretraining Model ===")
        
        # If generating mock data was requested
        if args.generate_mock:
            print(f"Generating {args.samples} mock workout records...")
            exercises = args.exercises.split(',') if args.exercises else None
            self.data_collector.generate_and_save_mock_data(
                num_samples=args.samples,
                exercises=exercises
            )
        
        # If importing from a CSV file was requested
        if args.import_file:
            if not os.path.exists(args.import_file):
                print(f"Error: File not found: {args.import_file}")
                return
                
            print(f"Importing data from {args.import_file}...")
            self.data_collector.import_from_csv(args.import_file, is_pretraining=True)
        
        # Specify model type if provided
        if args.model_type:
            print(f"Using {args.model_type} model type for training.")
            result = self.predictor.switch_model_type(args.model_type)
            if not result['success']:
                print(f"Error switching model type: {result['message']}")
                return
        
        # Load the training data
        print("Loading training data...")
        training_data = self.data_collector.load_training_data(include_pretraining=True)
        
        if training_data.empty:
            print("No training data available. Please generate mock data or import a CSV file first.")
            return
        
        # Train the model
        print(f"Pretraining model with {len(training_data)} workout records...")
        result = self.predictor.fit_model(training_data)
        
        if result['success']:
            print("Model pretraining complete!")
            print(f"Total training time: {result['training_time']:.2f} seconds")
        else:
            print(f"Model training failed: {result['message']}")
    
    def handle_collect(self, args: argparse.Namespace) -> None:
        """
        Handle the collect command to gather workout data interactively.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Collecting Workout Data ===")
        
        # Interactive data collection
        workout_data = self.data_collector.interactive_data_entry(args.exercise)
        
        # Save the data
        self.data_collector.save_workout_data(workout_data, is_pretraining=args.pretraining)
        
        # Ask if the user wants to add more data
        while input("\nAdd another workout? (y/n): ").lower().startswith('y'):
            workout_data = self.data_collector.interactive_data_entry(args.exercise)
            self.data_collector.save_workout_data(workout_data, is_pretraining=args.pretraining)
    
    def handle_interactive_training(self, args: argparse.Namespace) -> None:
        """
        Handle the interactive training command to train the model with user feedback.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Interactive Training ===")
        
        # Check if there's any data to work with
        training_data = self.data_collector.load_training_data(include_pretraining=True)
        
        if training_data.empty:
            print("No training data available. Please collect some data first.")
            return
        
        # Fit the model if not already trained
        print("Ensuring model is trained with existing data...")
        self.predictor.fit_model(training_data)
        
        # Exercise selection for prediction
        if args.exercise:
            exercise = args.exercise
        else:
            # List available exercises from the data
            available_exercises = training_data['exercise'].unique()
            print("\nAvailable exercises:")
            for i, ex in enumerate(available_exercises, 1):
                print(f"{i}. {ex}")
            
            # Let user select an exercise
            selection = input("\nSelect an exercise (number or name): ").strip()
            if selection.isdigit() and 1 <= int(selection) <= len(available_exercises):
                exercise = available_exercises[int(selection) - 1]
            elif selection in available_exercises:
                exercise = selection
            else:
                print("Invalid selection. Please enter a new exercise name:")
                exercise = input().strip()
        
        # Get previous workouts for this exercise
        exercise_data = training_data[training_data['exercise'] == exercise]
        
        if len(exercise_data) == 0:
            print(f"No previous data for {exercise}. Starting with a new exercise.")
            previous_workouts = []
        else:
            # Convert DataFrame rows to dictionaries
            previous_workouts = exercise_data.to_dict('records')
        
        # Make a prediction
        prediction_result = self.predictor.predict_workout(
            user_data={
                'exercise': exercise,
                'last_weight': exercise_data['weight'].iloc[-1] if not exercise_data.empty else 0
            }
        )
        
        print(f"\nPredicted weight for {exercise}: {prediction_result['predicted_weight']} kg/lb")
        print(f"Confidence: {prediction_result['confidence']:.2f}")
        
        if 'suggested_reps' in prediction_result:
            print(f"Suggested reps: {prediction_result['suggested_reps']}")
        
        if 'message' in prediction_result:
            print(f"Message: {prediction_result['message']}")
        
        if 'analysis' in prediction_result:
            print("\nAnalysis:")
            for key, value in prediction_result['analysis'].items():
                print(f"- {key}: {value}")
        
        # Ask if the user performed this workout
        if input("\nDid you perform this workout? (y/n): ").lower().startswith('y'):
            # Collect actual results
            print("\nEnter the actual results:")
            actual_weight = self.data_collector._get_validated_numeric_input("Actual weight used (kg/lb): ")
            reps = self.data_collector._get_validated_numeric_input("Reps completed: ", is_int=True)
            
            rir_input = input("RIR (Reps In Reserve, optional): ").strip()
            rir = int(rir_input) if rir_input and rir_input.isdigit() else None
            
            success = input("Was the workout successful? (y/n): ").lower().startswith('y')
            
            # Record feedback
            feedback = self.predictor.record_feedback(
                exercise=exercise,
                predicted_weight=prediction_result['predicted_weight'],
                actual_weight=actual_weight,
                success=success,
                reps=reps,
                rir=rir
            )
            
            print(f"\nFeedback recorded: {feedback['message']}")
            
            # Save this workout to the training data
            workout_data = {
                "exercise": exercise,
                "weight": actual_weight,
                "reps": reps,
                "sets": 1,  # Default to 1 set for simplicity
                "date": datetime.now().date().isoformat(),
                "rir": rir,
                "success": success
            }
            
            self.data_collector.save_workout_data(workout_data)
            
            # Make a new prediction with the updated data
            print("\nUpdating prediction with new feedback...")
            previous_workouts.append(workout_data)
            new_prediction = self.predictor.predict_workout(
                user_data={
                    'exercise': exercise,
                    'last_weight': actual_weight  # Use the actual weight as the last weight
                }
            )
            
            print(f"\nNext workout prediction for {exercise}: {new_prediction['predicted_weight']} kg/lb")
            print(f"Confidence: {new_prediction['confidence']:.2f}")
            if 'suggested_reps' in new_prediction:
                print(f"Suggested reps: {new_prediction['suggested_reps']}")
        
        print("\nInteractive training session complete!")
    
    def handle_predict(self, args: argparse.Namespace) -> None:
        """
        Handle the predict command to make a weight prediction.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Making a Prediction ===")
        
        # Load training data
        training_data = self.data_collector.load_training_data(include_pretraining=True)
        
        if training_data.empty:
            print("No training data available. Please collect some data first.")
            return
        
        # Ensure model is trained
        self.predictor.fit_model(training_data)
        
        # Get the exercise
        exercise = args.exercise
        
        if not exercise:
            # List available exercises
            available_exercises = training_data['exercise'].unique()
            print("\nAvailable exercises:")
            for i, ex in enumerate(available_exercises, 1):
                print(f"{i}. {ex}")
            
            # Let user select an exercise
            selection = input("\nSelect an exercise (number or name): ").strip()
            if selection.isdigit() and 1 <= int(selection) <= len(available_exercises):
                exercise = available_exercises[int(selection) - 1]
            elif selection in available_exercises:
                exercise = selection
            else:
                print("Invalid selection. Please enter a new exercise name:")
                exercise = input().strip()
        
        # Get previous workouts for this exercise
        exercise_data = training_data[training_data['exercise'] == exercise]
        
        if len(exercise_data) == 0:
            print(f"No previous data for {exercise}. Cannot make a prediction.")
            return
        
        # Convert DataFrame rows to dictionaries
        previous_workouts = exercise_data.to_dict('records')
        
        # Make a prediction, passing the debug flag
        prediction_result = self.predictor.predict_workout(
            user_data={
                'exercise': exercise,
                'last_weight': exercise_data['weight'].iloc[-1] if not exercise_data.empty else 0
            },
            debug=getattr(args, 'debug', False)  # Get the debug flag or default to False
        )
        
        print(f"\nPredicted weight for {exercise}: {prediction_result['predicted_weight']} kg/lb")
        print(f"Confidence: {prediction_result['confidence']:.2f}")
        
        if 'suggested_reps' in prediction_result:
            print(f"Suggested reps: {prediction_result['suggested_reps']}")
        
        if 'message' in prediction_result:
            print(f"Message: {prediction_result['message']}")
        
        if 'analysis' in prediction_result:
            print("\nAnalysis:")
            for key, value in prediction_result['analysis'].items():
                print(f"- {key}: {value}")
    
    def handle_reset(self, args: argparse.Namespace) -> None:
        """
        Handle the reset command to reset model data.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Resetting Model Data ===")
        
        reset_type = args.type if args.type else 'all'
        
        if reset_type not in ['all', 'model', 'feedback']:
            print(f"Invalid reset type: {reset_type}")
            print("Valid types are: all, model, feedback")
            return
        
        # Confirm reset
        if not args.yes:
            confirm = input(f"Are you sure you want to reset {reset_type} data? This cannot be undone. (y/n): ")
            if not confirm.lower().startswith('y'):
                print("Reset cancelled.")
                return
        
        # Perform the reset
        result = self.predictor.reset_model(reset_type)
        
        if result.get('success', False):
            print(f"Successfully reset {reset_type} data.")
        else:
            print(f"Failed to reset {reset_type} data: {result.get('message', 'Unknown error')}")
    
    def handle_export(self, args: argparse.Namespace) -> None:
        """
        Handle the export command to export training data.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Exporting Data ===")
        
        # Get the export file path
        file_path = args.file
        
        if not file_path:
            # Generate a default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"trainova_data_export_{timestamp}.csv"
        
        # Export the data
        success = self.data_collector.export_to_csv(
            file_path=file_path,
            include_pretraining=not args.exclude_pretraining
        )
        
        if success:
            print(f"Data successfully exported to {file_path}")
        else:
            print("Failed to export data.")
    
    def handle_import(self, args: argparse.Namespace) -> None:
        """
        Handle the import command to import workout data.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Importing Data ===")
        
        # Check if the file exists
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        
        # Import the data
        df = self.data_collector.import_from_csv(
            file_path=args.file,
            is_pretraining=args.pretraining
        )
        
        if not df.empty:
            print(f"Successfully imported {len(df)} records from {args.file}")
        else:
            print("Failed to import data.")
    
    def handle_evaluate(self, args: argparse.Namespace) -> None:
        """
        Handle the evaluate command to evaluate model performance.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Evaluating Model Performance ===")
        
        # Load training data
        training_data = self.data_collector.load_training_data(include_pretraining=True)
        
        if training_data.empty:
            print("No training data available. Please collect some data first.")
            return
        
        # Ensure model is trained
        self.predictor.fit_model(training_data)
        
        # Evaluate on the specified dataset
        if args.test_file:
            # Import test data if specified
            if not os.path.exists(args.test_file):
                print(f"Error: Test file not found: {args.test_file}")
                return
                
            print(f"Loading test data from {args.test_file}...")
            test_data = pd.read_csv(args.test_file)
        else:
            # Use existing data with train/test split
            print("Using existing data with 80/20 train/test split...")
            # Shuffle data to avoid time-based bias
            training_data = training_data.sample(frac=1, random_state=42)
            
            # Calculate split index (80% train, 20% test)
            split_idx = int(len(training_data) * 0.8)
            test_data = training_data.iloc[split_idx:]
        
        # Perform evaluation
        print(f"Evaluating model on {len(test_data)} test records...")
        result = self.predictor.evaluate_model(test_data)
        
        if result['success']:
            print("\nEvaluation Results:")
            for exercise, metrics in result['metrics'].items():
                print(f"\n{exercise}:")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        print(f"  {metric_name}:")
                        for k, v in metric_value.items():
                            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
                    else:
                        print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"  {metric_name}: {metric_value}")
        else:
            print(f"Evaluation failed: {result['message']}")
    
    def handle_switch_model(self, args: argparse.Namespace) -> None:
        """
        Handle the switch-model command to change the model type.
        
        Args:
            args: Command line arguments
        """
        print("\n=== Switching Model Type ===")
        
        # Get current model type
        current_model_type = self.predictor.model_type
        print(f"Current model type: {current_model_type}")
        
        # Get new model type
        new_model_type = args.model_type
        
        if not new_model_type:
            # Show available model types
            print("\nAvailable model types:")
            print("1. random_forest - Random Forest Regressor (fastest, good for small datasets)")
            print("2. neural_network - Neural Network (balanced speed/accuracy)")
            print("3. lstm - LSTM Neural Network (best for time series data, slowest)")
            
            # Let user select a model type
            selection = input("\nSelect a model type (number or name): ").strip()
            if selection == "1" or selection == "random_forest":
                new_model_type = "random_forest"
            elif selection == "2" or selection == "neural_network":
                new_model_type = "neural_network"
            elif selection == "3" or selection == "lstm":
                new_model_type = "lstm"
            else:
                print("Invalid selection. Using random_forest as default.")
                new_model_type = "random_forest"
        
        # Switch model type
        if new_model_type == current_model_type:
            print(f"Already using {new_model_type} model type.")
            return
        
        print(f"Switching to {new_model_type} model type...")
        result = self.predictor.switch_model_type(new_model_type)
        
        if result['success']:
            print(result['message'])
            
            # Ask if user wants to retrain
            if input("Do you want to retrain the model with the new model type? (y/n): ").lower().startswith('y'):
                print("\nRetraining model...")
                training_data = self.data_collector.load_training_data(include_pretraining=True)
                if not training_data.empty:
                    result = self.predictor.fit_model(training_data)
                    if result['success']:
                        print("Model training complete!")
                        print(f"Total training time: {result['training_time']:.2f} seconds")
                    else:
                        print(f"Model training failed: {result['message']}")
                else:
                    print("No training data available for retraining.")
        else:
            print(f"Failed to switch model type: {result['message']}")