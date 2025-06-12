import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run the Flask API server"""
    # Import Flask here to avoid any potential import issues
    from flask import Flask
    
    # Define host and port
    host = "0.0.0.0"
    port = 5010
    
    print(f"Starting Trainova Machine Learning API on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Dynamically import the Flask app and run it
    from trainova_ml.src.api.main import app
    app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
    main()