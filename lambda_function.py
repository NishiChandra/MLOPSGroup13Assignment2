import boto3
import joblib
import json
import os

# Initialize S3 client
s3 = boto3.client('s3')

# Bucket and file details
BUCKET_NAME = 'mlopsdiabetes'
MODEL_FILE_NAME = 'model.pkl'

def download_model_from_s3(bucket_name, model_file_name):
    # Create local path to store model
    local_model_path = f'/tmp/{model_file_name}'
    
    # Download the model from S3 to /tmp/ directory
    s3.download_file(bucket_name, model_file_name, local_model_path)
    
    return local_model_path

def load_model(local_model_path):
    # Load the model using joblib
    with open(local_model_path, 'rb') as model_file:
        model = joblib.load(model_file)
    return model

def handler(event, context):
    try:
        # Step 1: Download the model from S3
        local_model_path = download_model_from_s3(BUCKET_NAME, MODEL_FILE_NAME)
        
        # Step 2: Load the TPOT model
        model = load_model(local_model_path)
        
        # Step 3: Parse input data from event
        body = json.loads(event['body'])
        input_data = body.get('input_data')
        
        # Step 4: Make predictions
        prediction = model.predict(input_data)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction.tolist()  # Convert prediction to list for JSON serialization
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
