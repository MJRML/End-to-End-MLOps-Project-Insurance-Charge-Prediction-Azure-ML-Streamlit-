"""
--------
Scoring script for Azure ML Managed Online Endpoint.

Azure ML will:
1. Call init() once when the container starts:
      - We load the trained model (model.pkl).
2. Call run(data) for every incoming request:
      - We parse the JSON payload.
      - Convert it into a pandas DataFrame.
      - Call model.predict(...) and return results.
"""

import json
import joblib
import pandas as pd
import os

# Global variable to hold the model in memory
model = None

def init():
    """
    init() is called once when the deployment starts.
    We load model.pkl from the working directory.
    """
    global model

    # Azure ML mounts the model file into the working directory.
    # Usually the path is just "model.pkl" inside the model folder.
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "model.pkl")

    # If the above doesn't work in your setup, you can also try just "model.pkl"
    # model_path = "model.pkl"

    # Load trained pipeline (preprocessor + XGBoost)
    model = joblib.load(model_path)


def run(raw_data):
    """
    run() is called for every request.

    Expected input format (JSON):

    {
        "data": [
            {
                "age": 25,
                "sex": "male",
                "bmi": 27.9,
                "children": 0,
                "smoker": "yes",
                "region": "southwest"
            },
            {
                "age": 40,
                "sex": "female",
                "bmi": 30.1,
                "children": 2,
                "smoker": "no",
                "region": "northwest"
            }
        ]
    }

    
      - Parse JSON
      - Build a DataFrame
      - Call model.predict
      - Return predictions as JSON
    """
    try:
        # Parse the incoming JSON string
        data = json.loads(raw_data)

        # "data" should be a list of rows (dicts)
        records = data.get("data", [])

        # Convert to DataFrame with the SAME columns as training
        df = pd.DataFrame(records, columns=["age", "sex", "bmi", "children", "smoker", "region"])

        # Use our trained pipeline (preprocessor + regressor)
        preds = model.predict(df)

        # Convert numpy array to list for JSON serialization
        result = {"predictions": preds.tolist()}

        # Return as JSON string
        return json.dumps(result)

    except Exception as e:
        # If anything goes wrong, return a useful error
        return json.dumps({"error": str(e)})
