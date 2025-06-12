from flask import Flask, request, jsonify
import os
import sys
import json

# Add the main package directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trainova_ml.src.prediction.predictor import MLWorkoutPredictor

app = Flask(__name__)

# Initialize the predictor with neural network model
predictor = MLWorkoutPredictor(model_type='neural_network')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({
        'status': 'healthy',
        'model_type': predictor.model_type
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction for the next workout weight."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = ['exercise', 'last_weight']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        result = predictor.predict_workout(data)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Record user feedback about a prediction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = ['exercise', 'predicted_weight', 'actual_weight', 'success']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Optional fields
        reps = data.get('reps')
        rir = data.get('rir')
        
        # Record feedback
        result = predictor.record_feedback(
            exercise=data['exercise'],
            predicted_weight=float(data['predicted_weight']),
            actual_weight=float(data['actual_weight']),
            success=bool(data['success']),
            reps=reps,
            rir=rir
        )
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch-model', methods=['POST'])
def switch_model():
    """Switch to a different model type."""
    try:
        data = request.get_json()
        if not data or 'model_type' not in data:
            return jsonify({'error': 'No model_type provided'}), 400
        
        new_model_type = data['model_type']
        if new_model_type not in ['random_forest', 'neural_network', 'lstm']:
            return jsonify({'error': f'Invalid model type: {new_model_type}'}), 400
        
        result = predictor.switch_model_type(new_model_type)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate the model with test data."""
    try:
        # Check if the request has a file
        if 'file' not in request.files:
            return jsonify({'error': 'No test data file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the uploaded file temporarily
        temp_file_path = os.path.join(os.path.dirname(__file__), 'temp_test_data.csv')
        file.save(temp_file_path)
        
        # Read the CSV into a DataFrame
        import pandas as pd
        test_data = pd.read_csv(temp_file_path)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        # Evaluate the model
        result = predictor.evaluate_model(test_data)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Train the model using CSV data."""
    try:
        # Check if the request has a file
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            # Save the uploaded file temporarily
            temp_file_path = os.path.join(os.path.dirname(__file__), 'temp_training_data.csv')
            file.save(temp_file_path)
            
            # Read the CSV into a DataFrame
            import pandas as pd
            training_data = pd.read_csv(temp_file_path)
            
            # Clean up the temporary file
            os.remove(temp_file_path)
        elif request.json and 'use_sample_data' in request.json and request.json['use_sample_data']:
            # Use sample/pretraining data
            sample_data_path = os.path.join('trainova_ml', 'data', 'datasets', 'pretraining_data.csv')
            
            if not os.path.exists(sample_data_path):
                return jsonify({'error': 'Sample training data not found'}), 404
                
            training_data = pd.read_csv(sample_data_path)
        else:
            return jsonify({'error': 'No training data provided. Either upload a CSV file or set use_sample_data to true'}), 400
            
        # Get the model type from the request if specified
        if request.json and 'model_type' in request.json:
            model_type = request.json['model_type']
            if model_type not in ['random_forest', 'neural_network', 'lstm']:
                return jsonify({'error': f'Invalid model type: {model_type}'}), 400
                
            # Switch to the requested model type if different from current
            if model_type != predictor.model_type:
                predictor.switch_model_type(model_type)
        
        # Train the model
        result = predictor.fit_model(training_data)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load environment variables
    port = int(os.environ.get('PORT', 5010))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port, debug=debug)