"""
Trainova ML - Model Utilities
"""
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

def calculate_training_time(start_time: datetime) -> str:
    """
    Calculate and format the training time.
    
    Args:
        start_time: Start time of training
        
    Returns:
        Formatted time string
    """
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Format duration
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.2f}s"

def calculate_one_rep_max(weight: float, reps: int) -> float:
    """
    Calculate estimated one-rep max using the Brzycki formula.
    
    Args:
        weight: Weight lifted
        reps: Number of reps performed
        
    Returns:
        Estimated one-rep max
    """
    if reps <= 0:
        return weight
    
    if reps == 1:
        return weight
    
    # Brzycki formula: weight * (36 / (37 - reps))
    return weight * (36 / (37 - min(reps, 36)))

def calculate_weight_for_reps(one_rep_max: float, target_reps: int) -> float:
    """
    Calculate weight for target reps based on one-rep max.
    
    Args:
        one_rep_max: One-rep max
        target_reps: Target number of reps
        
    Returns:
        Weight for target reps
    """
    if target_reps <= 0:
        return one_rep_max
    
    if target_reps == 1:
        return one_rep_max
    
    # Inverse Brzycki formula
    weight = one_rep_max * ((37 - target_reps) / 36)
    return weight

def extract_time_features(date: pd.Series) -> pd.DataFrame:
    """
    Extract time-related features from date.
    
    Args:
        date: Series of dates
        
    Returns:
        DataFrame with time features
    """
    # Convert to datetime if string
    if date.dtype == 'object':
        date = pd.to_datetime(date)
    
    # Extract time features
    features = pd.DataFrame({
        'day_of_week': date.dt.dayofweek,
        'day_of_month': date.dt.day,
        'week_of_year': date.dt.isocalendar().week,
        'month': date.dt.month,
        'is_weekend': (date.dt.dayofweek >= 5).astype(int)
    })
    
    return features

def calculate_rest_days(dates: pd.Series) -> pd.Series:
    """
    Calculate days since last workout.
    
    Args:
        dates: Series of workout dates
        
    Returns:
        Series of rest days
    """
    # Convert to datetime if string
    if dates.dtype == 'object':
        dates = pd.to_datetime(dates)
    
    # Sort dates
    sorted_dates = dates.sort_values()
    
    # Calculate days between consecutive workouts
    rest_days = sorted_dates.diff().dt.days.fillna(0)
    
    return rest_days

def extract_workout_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from workout data for ML model.
    
    Args:
        data: DataFrame of workout data
        
    Returns:
        DataFrame with extracted features
    """
    # Ensure data is sorted by date
    if 'date' in data.columns:
        data = data.sort_values('date')
    
    # Create basic features
    features = pd.DataFrame()
    
    # Exercise features (one-hot encoding)
    if 'exercise' in data.columns:
        exercise_dummies = pd.get_dummies(data['exercise'], prefix='exercise')
        features = pd.concat([features, exercise_dummies], axis=1)
    
    # Weight features
    if 'weight' in data.columns:
        features['last_weight'] = data['weight'].shift(1).fillna(data['weight'].mean())
        features['weight_diff'] = data['weight'] - features['last_weight']
        features['weight_ratio'] = data['weight'] / features['last_weight'].replace(0, 1)
        
        # Rolling weight statistics
        features['weight_mean_3'] = data['weight'].rolling(3, min_periods=1).mean()
        features['weight_max_5'] = data['weight'].rolling(5, min_periods=1).max()
        features['weight_trend'] = data['weight'].rolling(3, min_periods=2).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        ).fillna(0)
    
    # Rep features
    if 'reps' in data.columns:
        features['last_reps'] = data['reps'].shift(1).fillna(data['reps'].mean())
        features['reps_diff'] = data['reps'] - features['last_reps']
        features['reps_ratio'] = data['reps'] / features['last_reps'].replace(0, 1)
        features['reps_mean_3'] = data['reps'].rolling(3, min_periods=1).mean()
    
    # Volume features (weight * reps)
    if 'weight' in data.columns and 'reps' in data.columns:
        data['volume'] = data['weight'] * data['reps']
        features['last_volume'] = data['volume'].shift(1).fillna(data['volume'].mean())
        features['volume_diff'] = data['volume'] - features['last_volume']
        features['volume_ratio'] = data['volume'] / features['last_volume'].replace(0, 1)
        features['volume_mean_3'] = data['volume'].rolling(3, min_periods=1).mean()
    
    # Set features
    if 'sets' in data.columns:
        features['last_sets'] = data['sets'].shift(1).fillna(data['sets'].mean())
        features['sets_diff'] = data['sets'] - features['last_sets']
    
    # RIR (Reps In Reserve) features
    if 'rir' in data.columns:
        # Fill missing RIR with median
        median_rir = data['rir'].median()
        features['last_rir'] = data['rir'].shift(1).fillna(median_rir)
        features['rir_diff'] = data['rir'] - features['last_rir']
    
    # Time features
    if 'date' in data.columns:
        # Extract time features
        time_features = extract_time_features(data['date'])
        features = pd.concat([features, time_features], axis=1)
        
        # Calculate rest days
        features['rest_days'] = calculate_rest_days(data['date'])
    
    # Success rate features
    if 'success' in data.columns:
        features['last_success'] = data['success'].shift(1).fillna(1).astype(int)
        features['success_rate_3'] = data['success'].rolling(3, min_periods=1).mean()
    
    # One-rep max features
    if 'weight' in data.columns and 'reps' in data.columns:
        data['estimated_1rm'] = data.apply(
            lambda row: calculate_one_rep_max(row['weight'], row['reps']), axis=1
        )
        features['last_1rm'] = data['estimated_1rm'].shift(1).fillna(data['estimated_1rm'].mean())
        features['1rm_diff'] = data['estimated_1rm'] - features['last_1rm']
        features['1rm_ratio'] = data['estimated_1rm'] / features['last_1rm'].replace(0, 1)
        features['1rm_mean_3'] = data['estimated_1rm'].rolling(3, min_periods=1).mean()
        features['1rm_max_5'] = data['estimated_1rm'].rolling(5, min_periods=1).max()
        features['1rm_trend'] = data['estimated_1rm'].rolling(3, min_periods=2).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        ).fillna(0)
    
    # Drop any NaN values that might have been introduced
    features = features.fillna(0)
    
    return features

def generate_feedback_message(score: float) -> str:
    """
    Generate feedback message based on score.
    
    Args:
        score: Feedback score (-1 to 1)
        
    Returns:
        Feedback message
    """
    if score > 0.5:
        return "Great job! Your performance exceeded expectations. Keep up the good work!"
    elif score > 0.2:
        return "Good work! You're making solid progress. Keep pushing yourself."
    elif score > -0.2:
        return "Decent effort. The prediction was fairly accurate for your performance level."
    elif score > -0.5:
        return "The weight might have been a bit challenging. Consider adjusting your training intensity."
    else:
        return "This workout was too challenging. Let's adjust your weights for better results next time."