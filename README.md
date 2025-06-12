# Trainova Machine Learning

A machine learning-based workout prediction system that uses actual ML models to predict workout weights and improve through user feedback.

## Overview

Trainova ML is an advanced version of the original Trainova Feedback Network that implements proper machine learning models to predict workout weights. The system can train on historical workout data, make intelligent predictions, and continuously improve through user feedback.

Key features:

- Multiple ML model types (Random Forest, Neural Network, LSTM)
- Realistic training times proportional to dataset size
- Feature engineering from workout data
- Interactive prediction and feedback loop
- Mock data generation for model pretraining
- Performance evaluation and model comparison

## Installation

### Option 1: Standard Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/trainova_machine_learning_v2.git
cd trainova_machine_learning_v2
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/trainova_machine_learning_v2.git
cd trainova_machine_learning_v2
```

2. Build and run using Docker Compose:

```bash
docker-compose build
docker-compose up
```

3. Run specific commands using Docker:

```bash
# Basic command format
docker-compose run trainova [command] [options]

# Examples:
docker-compose run trainova predict --exercise "Bench Press"
docker-compose run trainova pretrain --generate-mock --samples 1000
docker-compose run trainova evaluate
```

4. Stop the container:

```bash
docker-compose down
```

The Docker setup includes volume mappings for the datasets and models folders, ensuring your data persists between container runs.

## Usage

Trainova ML provides a command-line interface with several commands:

### Pretraining the Model

Generate mock data and pretrain the model:

```bash
python trainova-cli.py pretrain --generate-mock --samples 1000 --model-type random_forest
```

Pretrain using your own CSV data:

```bash
python trainova-cli.py pretrain --import-file your_data.csv
```

### Interactive Training

Start an interactive training session:

```bash
python trainova-cli.py interactive
```

This will:

1. Make a prediction for your next workout
2. Allow you to enter actual results
3. Record feedback and improve future predictions

### Making Predictions

Get a weight prediction for your next workout:

```bash
python trainova-cli.py predict --exercise "Bench Press"
```

### Collecting Workout Data

Manually record workout data:

```bash
python trainova-cli.py collect
```

### Importing/Exporting Data

Import workout data from a CSV:

```bash
python trainova-cli.py import workout_data.csv
```

Export your data:

```bash
python trainova-cli.py export --file my_workouts.csv
```

### Evaluating Model Performance

Evaluate the model's performance:

```bash
python trainova-cli.py evaluate
```

### Switching Model Types

Change the machine learning model type:

```bash
python trainova-cli.py switch-model --model-type neural_network
```

## Machine Learning Models

Trainova ML includes multiple model types:

1. **Random Forest** (`random_forest`): Fast training, good for small datasets, handles categorical features well.

2. **Neural Network** (`neural_network`): Balanced training speed and accuracy, can capture complex patterns.

3. **LSTM** (`lstm`): Best for time-series workout data, captures temporal patterns in your training progression.

## Feature Engineering

The system automatically extracts features from your workout data:

- Exercise-specific patterns
- Weight progression trends
- Rep and volume patterns
- Rest days analysis
- Success rate metrics
- Estimated 1RM calculations

## Expected Training Times

Unlike the original Trainova Feedback Network, this machine learning version has realistic training times that scale with dataset size:

- 1,000 records: ~5-10 seconds (Random Forest), ~30-60 seconds (Neural Network)
- 10,000 records: ~1-2 minutes (Random Forest), ~5-10 minutes (Neural Network)
- 100,000 records: ~10-20 minutes (Random Forest), ~1-2 hours (Neural Network)
- 1,000,000 records: ~1-2 hours (Random Forest), ~10-24 hours (Neural Network)

## Project Structure

```
trainova_machine_learning_v2/
├── requirements.txt        # Project dependencies
├── trainova-cli.py         # CLI entry point
├── trainova_ml/            # Main package
│   ├── __init__.py
│   ├── data/               # Data handling
│   │   ├── __init__.py
│   │   └── data_collection.py
│   ├── src/                # Source code
│   │   ├── __init__.py
│   │   ├── cli/            # Command-line interface
│   │   │   ├── __init__.py
│   │   │   ├── commands.py
│   │   │   └── main.py
│   │   ├── models/         # ML model implementations
│   │   │   ├── __init__.py
│   │   │   └── ml_models.py
│   │   ├── prediction/     # Prediction logic
│   │   │   ├── __init__.py
│   │   │   └── predictor.py
│   │   └── utils/          # Utility functions
│   │       ├── __init__.py
│   │       └── model_utils.py
│   └── data/               # Data storage
│       ├── datasets/       # CSV datasets
│       └── models/         # Saved models
```

## API Usage

Trainova ML now provides a RESTful API to access the neural network model via HTTP requests, making it easier to integrate with other applications.

### Running the API

With Docker:

```bash
# Start the API server
docker-compose up

# To stop the API server
docker-compose down
```

Without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

### API Endpoints

#### Health Check

```
GET /health
```

Verify the API is running and which model type is being used.

#### Predict Workout

```
POST /predict
```

Make a prediction for the next workout weight.

Example request:

```json
{
  "exercise": "Bench Press",
  "last_weight": 70,
  "reps": 5,
  "intensity": 8
}
```

#### Record Feedback

```
POST /feedback
```

Record user feedback about a prediction.

Example request:

```json
{
  "exercise": "Bench Press",
  "predicted_weight": 72.5,
  "actual_weight": 70,
  "success": true,
  "reps": 5,
  "rir": 2
}
```

#### Switch Model

```
POST /switch-model
```

Switch to a different model type.

Example request:

```json
{
  "model_type": "lstm"
}
```

#### Evaluate Model

```
POST /evaluate
```

Evaluate the model with test data. Send a CSV file with the field name 'file'.

### Using the CLI with Docker

With the new API setup, you can still use the CLI tool:

```bash
# Run CLI commands with Docker
docker-compose --profile cli run trainova-cli predict --exercise "Bench Press"
docker-compose --profile cli run trainova-cli pretrain --generate-mock --samples 1000
docker-compose --profile cli run trainova-cli evaluate
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
