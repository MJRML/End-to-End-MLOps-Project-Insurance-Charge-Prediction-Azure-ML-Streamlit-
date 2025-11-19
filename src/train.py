'''

Training script for insurace cost prediction

--------------------------------------------

The script is deigned torun both locally and inside AZure ML

1. read the insurance csv dataset  (path provided with --data-path)
2: splits the data into features and target column
3: Builds a preprocessing pipeline:
        - One-hot encoding categorical columns
        - Keeps numeric columnsas is
4: Trains an XGBoost regression model
5: Evalutes the model using MSE
6: Logs the metric back into azure ml
7: Saves the trained model to a model.pkl file
8: Uploads the model.pkl into Azure ML runs 'outputs/' folder
'''


import argparse  #for parsing --data-path from azure ml job
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from azureml.core import Run  #used to communicate back to azure ml


#get the azure ml conntext (if running locally, this become a dummy run)

run = Run.get_context()

def main(args):
    '''
    Main training funtion
    '''
    
    #load the dataset
    #args.data_path comes from argparse (azure ml passes it)
    df = pd.read_csv(args.data_path)
    
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    #define column types
    categorical = ['sex','smoker','region'] #string categories
    numeric = ['age','bmi','children'] #numeric values
    
    #define the preprocessing pipeline
    #one hotencoder converts text categories --> binary vectors
    #'passthrough' means keep numeric columns unchanged
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('num', 'passthrough', numeric)
        ]
    )
    
    #Build the model pipeline
    #Pipeline ensures preprocessing happens before modelling
    
    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=200, #number of trees
                learning_rate=0.1, #step_size
                max_depth=6  #tree depth
            ))
        ]
    )
    
    #Train/test Split
    
    X_train,X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=99
    )
    
    #Train the model
    model.fit(X_train, y_train)
    
    #evaluate the model
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    #log mse ito Azure ML so it appears in the experiment UI
    run.log('mse',mse)
    
    #save the trained model locally
    #azure mol will automatically grab anythin uploaded to outputs/
    joblib.dump(model,'model.pkl')
    
    #upload to azure ml run artifacts
    run.upload_file('outputs/model.pkl', 'model.pkl')
    
    #mark run complete
    run.complete()
    
#Entry point  - executes when script runs
if __name__ == '__main__':
    #arg parse --data-patrh value passed in from azure ml
    parser = argparse.ArgumentParser(description='Train insurance ML Model')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to insurance.csv inside Azure ML')
    args = parser.parse_args()
    
    #Start training
    main(args)