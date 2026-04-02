from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import requests 
from io import BytesIO

app = FastAPI()


# load model once at startup
# using Hugging face: 
# "text-classification" tells HuggingFace we want to classify text
# "distilbert-base-uncased-finetuned-sst-2-english" is a pretrained model that classifies texts as positive or negative
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print("Model loaded!")


# input/output schemas

# defines what the user must send TO /predict
class PredictRequest(BaseModel):
    text: str

# what the response FROM /predict looks like
# label is either "POSITIVE" or "NEGATIVE"
# score is the model's confidence between 0 and 1
class PredictResponse(BaseModel): 
    label: str
    score: float

# endpoints

# main endpoint: confirms the API is running
@app.get('/')
def main():
	return {'message': 'Text Classification API is running'}

# health endpointL confirms the model loaded successfully
@app.get('/health')
def health():
     if classifier is not None:
        return {"status": "healthy", "model": "distilbert-base-uncased-finetuned-sst-2-english"}
     else:
        raise HTTPException(status_code=500, detail="Model not loaded")
     
# predict endpoint 
@app.post('/predict', response_model = PredictResponse)
def predict(request: PredictRequest):
    # make sure user sent text
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text can't be empty")
    
    # run text through model
    result = classifier(request.text)

    return PredictResponse(label=result[0]["label"], score=round(result[0]["score"],4))