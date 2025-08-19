from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("femnaja/scam")
model = AutoModelForSequenceClassification.from_pretrained("femnaja/scam")
model.eval()

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(msg: Message):
    inputs = tokenizer(msg.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return {"scam": bool(prediction)}

