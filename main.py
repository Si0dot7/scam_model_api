from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ ให้ CORSMiddleware จัดการเอง
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

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(msg: Message):
    inputs = tokenizer(msg.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    scam_prob = probs[0][1].item()
    return {
        "scam": scam_prob > 0.5,
        "probability": scam_prob
    }
