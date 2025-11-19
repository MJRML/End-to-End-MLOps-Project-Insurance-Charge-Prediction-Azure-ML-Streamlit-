# End-to-End MLOps Project: Insurance Charge Prediction (Azure ML + Streamlit)

This project demonstrates a full **MLOps workflow** using **Azure Machine Learning**, including:

- Reproducible model training with Azure ML Jobs  
- Model versioning in the Azure ML Model Registry  
- Custom inference environment  
- Managed Online Endpoint deployment  
- Real-time predictions via REST API  
- A Streamlit web frontend for user interaction  

This mirrors real-world enterprise MLOps frameworks used in production.

---

##  Tech Stack

### **Machine Learning**
- XGBoost Regression
- Scikit-learn
- Python 3.10

### **Azure Cloud**
- Azure Machine Learning
- Azure Managed Online Endpoints
- Azure ML Model Registry
- Azure ML Environments
- Azure Compute Clusters

### **Frontend**
- Streamlit  
- REST API integration  
- Bearer token authentication  

### Azure Resources Used

- Resource Group
- Azure Machine Learning Workspace
- Azure Storage Account (Blob)
- Azure ML Datastore
- Azure Compute Cluster (cpu-cluster)
- Azure ML Environment
- Azure ML Model Registry
- Azure Managed Online Endpoint
- Azure Online Deployment (blue)

---

## 📁 Project Structure

├── src/  
│ ├── train.py  #Training script  
│ ├── score.py  #Inference script  
│  
├── jobs/  
│ └── job-train.yml  #Azure ML training job definition   
│    
├── environment/  
│ └── environment.yml  #Conda environment for inference  
│  
├── deployment/  
│ └── deployment.yml  #Online endpoint deployment YAML  
│  
├── streamlit_app/  
│ └── app.py  #Streamlit frontend  
│  
├── test/  
│ └── test.json  #Sample request for testing endpoint  
│  
├── requirements.txt  
└── README.md  



---

# Step 1 — Local Training Setup (Reproducible ML)

- Project structure is created (src/, jobs/, environment/, etc.)
- train.py is implemented with:
- Reading the insurance dataset
- Preprocessing features
- Training an XGBoost regression model
- Calculating metrics (MSE, RMSE, R²)
- Saving the trained model as model.pkl
- Script uses argparse, so it works locally and in Azure ML

```bash
python src/train.py --data-path data/insurance.csv
```

# Step 2 — Train the Model on Azure ML (CommandJob)

- **Azure ML provides:**
- **Scalable compute**
- **Centralized logs & metrics**
- **Reproducible runs**
- **Automated artifact storage**
- **This step represents the training pipeline in MLOps**  

- A Command Job YAML (job-train.yml) defines:
- Compute cluster to run training
- The command: python train.py --data-path ...
- Inputs (insurance dataset in Blob Storage)
- Environment (conda + Docker image)
- Output artifacts (model.pkl saved to Azure storage)  

```bash
az ml job create -f jobs/job-train.yml -g ml-rg -w ml-ws  
```

# Step 3 — Register the Trained Model (Model Registry)  (You can register the model from GUI)

- After the training job completes, Azure ML stores model.pkl
- That model is registered as: **insurance-xgb-model:1**


- **The registry stores:**  
- Model file
- Metadata
- Training lineage
- Version number

```bash
az ml model create \
  --name insurance-xgb-model \
  --path runs:/<run-id>/model.pkl \
  -g ml-rg -w ml-ws
```

# Step 4 — Create the Inference Environment

- **environment.yml defines:**
- Python version (3.10)
- Dependencies:
- pandas
- xgboost
- scikit-learn
- joblib
- azureml-inference-server-http  

- **Base Docker image:**
```bash
mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
```

```bash
az ml environment create \
  --file environment/environment.yml \
  -g ml-rg -w ml-ws
```

# Step 5 — Deploy Model to a Managed Online Endpoint

- Creates the API “container”
- Generates scoring URL
- Generates primary and secondary API keys
- Sets up logging, monitoring, and security

```bash
az ml online-endpoint create \
  --name insurance-endpoint \
  --auth-mode key \
  -g ml-rg -w ml-ws
```

# Step 5.2 Create the Deployment (Blue)

- **The deployment includes:**
- Model (insurance-xgb-model:1)
- Scoring script (score.py)
- Environment (insurance-serving-env:1)
- Compute instance type (Standard_DS2_v2)
- Traffic routing (100% → blue)


```bash
az ml online-deployment create \
  --name blue \
  --endpoint-name insurance-endpoint \
  --file deployment/deployment.yml \
  -g ml-rg -w ml-ws \
  --all-traffic
```

# Step 5.3 Test the Live Endpoint

```json
{
  "data": [
    {
      "age": 25,
      "sex": "male",
      "bmi": 28,
      "children": 1,
      "smoker": "no",
      "region": "southwest"
    }
  ]
}
```

- **INVOKE**
```bash
az ml online-endpoint invoke \
  --name insurance-endpoint \
  --request-file test/test.json \
  -g ml-rg -w ml-ws
```
# Step 7 — Getting Your Azure ML Endpoint Keys (for streamlit UI )

- Azure ML uses key-based authentication for Managed Online Endpoints.  

- **R**etrieve Endpoint Keys**  
```bash
az ml online-endpoint get-credentials \
  --name insurance-endpoint \
  --resource-group ml-rg \
  --workspace-name ml-ws
```

- **Example Ouput**
- Use only the primaryKey in Streamlit   
```json
{
  "primaryKey": "xxxx",
  "secondaryKey": "yyyy"
}
```


# Step 7 — Build the Streamlit Frontend (User Interface)

- The Streamlit app provides a simple UI so users can generate predictions without touching Azure ML directly.

- **How the Streamlit App Works:**
- User fills in details (age, BMI, smoker, region, etc.)
- Streamlit sends a POST request to the Azure ML Online Endpoint
- The endpoint runs your model and returns predictions
- The UI displays the output with proper formatting
- http://localhost:8501

- Run locally

```bash
streamlit run streamlit_app/app.py
```



# Future Improvements

- Add GitHub Actions CI/CD
- Add data drift monitoring (Azure ML)
- Add scheduled retraining
- Deploy Streamlit to Azure Web App
- Enable blue/green traffic shifting
