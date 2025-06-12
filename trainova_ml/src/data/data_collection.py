"""
Trainova ML - Data Collection Module
"""
import os
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional, Union

class DataCollector:
    """
    Handles data collection, manipulation, and storage for the ML-based Trainova system.
    This class provides methods to collect and preprocess training data.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data collector with a specific data directory.
        
        Args:
            data_dir: Directory to store collected data
        """
        # Set default data directory if none provided
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.datasets_dir = os.path.join(self.data_dir, "datasets")
        
        # Create directories if they don't exist
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Set default file paths
        self.training_data_path = os.path.join(self.datasets_dir, "training_data.csv")
        self.pretraining_data_path = os.path.join(self.datasets_dir, "pretraining_data.csv")
    
    def interactive_data_entry(self, exercise_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect workout data interactively from the user via CLI prompts.
        
        Args:
            exercise_type: Optional predefined exercise type
            
        Returns:
            Dictionary with collected workout data
        """
        print("\n=== Enter Workout Data ===")
        
        # Get exercise if not provided
        if not exercise_type:
            exercise_type = input("Exercise name: ").strip()
        else:
            print(f"Exercise: {exercise_type}")
        
        # Get required numeric inputs with validation
        weight = self._get_validated_numeric_input("Weight (kg/lb): ")
        reps = self._get_validated_numeric_input("Reps: ", is_int=True)
        sets = self._get_validated_numeric_input("Sets: ", is_int=True, default=1)
        
        # Get optional inputs
        rir_input = input("RIR (Reps In Reserve, optional): ").strip()
        rir = int(rir_input) if rir_input and rir_input.isdigit() else None
        
        rpe_input = input("RPE (Rate of Perceived Exertion 1-10, optional): ").strip()
        rpe = float(rpe_input) if rpe_input and self._is_numeric(rpe_input) else None
        
        # Ask if workout was successful
        success_input = input("Was the workout successful? (y/n): ").strip().lower()
        success = success_input.startswith('y')
        
        # Get the date (default to today)
        date_input = input("Date (YYYY-MM-DD, leave blank for today): ").strip()
        if date_input:
            try:
                date = datetime.strptime(date_input, "%Y-%m-%d").date()
            except ValueError:
                print("Invalid date format. Using today's date.")
                date = datetime.now().date()
        else:
            date = datetime.now().date()
        
        # Compile the workout data
        workout_data = {
            "exercise": exercise_type,
            "weight": weight,
            "reps": reps,
            "sets": sets,
            "date": date.isoformat(),
            "rir": rir,
            "rpe": rpe,
            "success": success
        }
        
        print("\nRecorded workout data:")
        for key, value in workout_data.items():
            if value is not None:
                print(f"{key}: {value}")
        
        return workout_data
    
    def _get_validated_numeric_input(self, prompt: str, is_int: bool = False, default: Optional[float] = None) -> Union[int, float]:
        """
        Get and validate numeric input from the user.
        
        Args:
            prompt: Input prompt to display
            is_int: Whether the input should be an integer
            default: Default value to use if input is empty
            
        Returns:
            Validated numeric value
        """
        while True:
            value_input = input(prompt).strip()
            
            # Use default if input is empty and default is provided
            if not value_input and default is not None:
                return default
                
            # Validate the input
            if self._is_numeric(value_input):
                value = float(value_input)
                if is_int and value.is_integer():
                    return int(value)
                elif is_int:
                    print("Please enter an integer value.")
                else:
                    return value
            else:
                print("Please enter a valid number.")
    
    def _is_numeric(self, value: str) -> bool:
        """
        Check if a string represents a valid numeric value.
        
        Args:
            value: String to check
            
        Returns:
            True if the string is a valid number, False otherwise
        """
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def save_workout_data(self, workout_data: Dict[str, Any], is_pretraining: bool = False) -> str:
        """
        Save workout data to the appropriate CSV file.
        
        Args:
            workout_data: Dictionary containing workout data
            is_pretraining: Whether this data is for pretraining
            
        Returns:
            Path to the saved file
        """
        # Determine the file path
        file_path = self.pretraining_data_path if is_pretraining else self.training_data_path
        
        # Check if the file exists to determine if we need to write headers
        file_exists = os.path.isfile(file_path)
        
        # Ensure all values are JSON serializable
        for key, value in list(workout_data.items()):
            if isinstance(value, (datetime, pd.Timestamp)):
                workout_data[key] = value.isoformat()
        
        # Get the fieldnames from the workout data
        fieldnames = list(workout_data.keys())
        
        # Write to the CSV file
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(workout_data)
        
        print(f"Data saved to {file_path}")
        return file_path
    
    def generate_mock_data(self, num_samples: int = 100, exercises: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate mock workout data for training ML models with realistic progression rates.
        
        Args:
            num_samples: Number of samples to generate
            exercises: List of exercises to generate data for
            
        Returns:
            DataFrame containing the generated data
        """
        if exercises is None:
            exercises = ["Squat", "Bench Press", "Deadlift", "Overhead Press", "Barbell Row"]
        
        # Lists to store the generated data
        data_rows = []
        
        # Generate data for each exercise
        for exercise in exercises:
            # Generate a reasonable starting weight for each exercise
            if exercise == "Squat":
                base_weight = random.uniform(60, 100)
            elif exercise == "Bench Press":
                base_weight = random.uniform(40, 80)
            elif exercise == "Deadlift":
                base_weight = random.uniform(80, 120)
            elif exercise == "Overhead Press":
                base_weight = random.uniform(30, 50)
            else:  # Default for other exercises
                base_weight = random.uniform(40, 70)
            
            # Generate samples per exercise
            samples_per_exercise = num_samples // len(exercises)
            
            # Generate a date range starting from 6 months ago
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=180)
            date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
            
            # Ensure we have enough dates
            if len(date_range) < samples_per_exercise:
                date_range = date_range * (samples_per_exercise // len(date_range) + 1)
            
            # Shuffle and select dates
            random.shuffle(date_range)
            workout_dates = sorted(date_range[:samples_per_exercise])
            
            # Progressive overload simulation with more realistic progression
            current_weight = base_weight
            
            # Approximate training frequency (workouts per week)
            workouts_per_week = len(workout_dates) / 26  # 26 weeks in 6 months
            
            # Calculate a realistic progression rate (based on 5-10kg per year for most lifts)
            # This would be roughly 0.1-0.2kg per week, or ~0.05kg per workout
            progression_per_workout = random.uniform(0.05, 0.1)
            
            # Track consecutive workouts with the same weight to simulate plateaus
            same_weight_count = 0
            
            # Add noise to simulate real-world variability
            noise_level = 0.05  # 5% noise
            
            for i in range(samples_per_exercise):
                # Slow, realistic progression
                if i > 0:
                    # Add progressive overload less frequently (15% chance)
                    if random.random() < 0.15:
                        # Most increases will be 2.5kg
                        current_weight += 2.5
                        same_weight_count = 0
                    # Small chance (5%) of decreasing weight (deload week)
                    elif random.random() < 0.05:
                        current_weight -= 2.5
                        same_weight_count = 0
                    else:
                        # No change in weight
                        same_weight_count += 1
                
                # Simulate plateaus being broken after several workouts at same weight
                if same_weight_count >= 5 and random.random() < 0.5:
                    current_weight += 2.5
                    same_weight_count = 0
                
                # Round to nearest 2.5
                current_weight = round(current_weight / 2.5) * 2.5
                
                # Ensure weight doesn't go below a minimum
                current_weight = max(current_weight, 20)
                
                # Generate reps, sets, RIR, RPE with more variation
                reps = random.randint(3, 12)
                sets = random.randint(1, 5)
                rir = random.randint(0, 4) if random.random() < 0.8 else None  # 80% chance of having RIR
                rpe = round(random.uniform(6, 10), 1) if random.random() < 0.6 else None  # 60% chance of having RPE
                
                # Add noise to simulate real-world variability
                # In real life, sometimes people will lift slightly more or less than planned
                noise_factor = 1 + random.uniform(-noise_level, noise_level)
                noisy_weight = current_weight * noise_factor
                
                # Round to nearest 1.0 for realism
                noisy_weight = round(noisy_weight)
                
                # Determine success rate (higher weights have lower success rates)
                # Base success rate of 90%, decreasing as weight increases
                base_success_rate = 0.9
                weight_factor = noisy_weight / base_weight
                success_rate = base_success_rate - (weight_factor - 1) * 0.3
                success_rate = max(0.5, min(success_rate, 0.95))  # Keep within reasonable range
                
                # Determine if workout was successful
                success = random.random() < success_rate
                
                # Create the data row with full feature set for ML training
                row = {
                    "exercise": exercise,
                    "weight": noisy_weight,
                    "reps": reps,
                    "sets": sets,
                    "date": workout_dates[i].isoformat(),
                    "rir": rir,
                    "rpe": rpe,
                    "success": success,
                    "day_of_week": workout_dates[i].weekday(),
                    "month": workout_dates[i].month,
                    "volume": noisy_weight * reps * sets
                }
                
                data_rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(data_rows)
        
        # Sort by date
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        return df
    
    def generate_and_save_mock_data(self, num_samples: int = 100, exercises: Optional[List[str]] = None) -> str:
        """
        Generate mock data and save it to the pretraining data file.
        
        Args:
            num_samples: Number of samples to generate
            exercises: List of exercises to generate data for
            
        Returns:
            Path to the saved file
        """
        df = self.generate_mock_data(num_samples, exercises)
        
        # Save to CSV
        df.to_csv(self.pretraining_data_path, index=False)
        
        print(f"Generated {len(df)} mock workout records and saved to {self.pretraining_data_path}")
        return self.pretraining_data_path
    
    def load_training_data(self, include_pretraining: bool = True) -> pd.DataFrame:
        """
        Load training data from CSV files.
        
        Args:
            include_pretraining: Whether to include pretraining data
            
        Returns:
            DataFrame containing the loaded data
        """
        # Initialize an empty list to store DataFrames
        dfs = []
        
        # Load training data if it exists
        if os.path.isfile(self.training_data_path):
            try:
                training_df = pd.read_csv(self.training_data_path)
                print(f"Loaded {len(training_df)} records from {self.training_data_path}")
                dfs.append(training_df)
            except Exception as e:
                print(f"Error loading training data: {e}")
        
        # Load pretraining data if requested and it exists
        if include_pretraining and os.path.isfile(self.pretraining_data_path):
            try:
                pretraining_df = pd.read_csv(self.pretraining_data_path)
                print(f"Loaded {len(pretraining_df)} records from {self.pretraining_data_path}")
                dfs.append(pretraining_df)
            except Exception as e:
                print(f"Error loading pretraining data: {e}")
        
        # Combine the DataFrames
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Convert date column to datetime
            if "date" in combined_df.columns:
                combined_df["date"] = pd.to_datetime(combined_df["date"])
                combined_df = combined_df.sort_values("date")
            
            return combined_df
        else:
            print("No data files found or all files were empty.")
            return pd.DataFrame()
    
    def import_from_csv(self, file_path: str, is_pretraining: bool = False) -> pd.DataFrame:
        """
        Import workout data from an external CSV file.
        
        Args:
            file_path: Path to the CSV file to import
            is_pretraining: Whether to save as pretraining data
            
        Returns:
            DataFrame containing the imported data
        """
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ["exercise", "weight", "reps"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Error: Missing required columns: {', '.join(missing_columns)}")
                return pd.DataFrame()
            
            # Ensure date column exists and is in the correct format
            if "date" not in df.columns:
                print("Warning: No date column found. Using current date for all entries.")
                df["date"] = datetime.now().date().isoformat()
            
            # Convert date to datetime
            df["date"] = pd.to_datetime(df["date"])
            
            # Save to the appropriate file
            target_path = self.pretraining_data_path if is_pretraining else self.training_data_path
            df.to_csv(target_path, index=False)
            
            print(f"Imported {len(df)} records from {file_path} and saved to {target_path}")
            return df
            
        except Exception as e:
            print(f"Error importing data: {e}")
            return pd.DataFrame()
    
    def export_to_csv(self, file_path: str, include_pretraining: bool = True) -> bool:
        """
        Export the current training data to a CSV file.
        
        Args:
            file_path: Path to save the CSV file
            include_pretraining: Whether to include pretraining data
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Load the data
            df = self.load_training_data(include_pretraining=include_pretraining)
            
            if df.empty:
                print("No data to export.")
                return False
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            print(f"Exported {len(df)} records to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False