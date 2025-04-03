import pandas as pd
import numpy as np
import uvicorn
import faiss
import torch
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
import json

# Initialize FastAPI
app = FastAPI(title="LLM-Powered Booking Analytics & QA System")

# Load data
df = pd.read_csv("hotel_bookings.csv")

# Data Preprocessing
df = df.copy()  # Avoid SettingWithCopyWarning
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce")

# ✅ Precompute analytics for efficiency
analytics_cache = {}

# Revenue trends over time
df["total_price"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]  # Approximation
revenue_trends = df.groupby("reservation_status_date")["total_price"].sum().to_dict()
analytics_cache["revenue_trends"] = revenue_trends

# Cancellation rate
cancellation_rate = (df["is_canceled"].sum() / len(df)) * 100
analytics_cache["cancellation_rate"] = round(cancellation_rate, 2)

# Geographical distribution of users
geo_distribution = df["country"].value_counts().to_dict()
analytics_cache["geo_distribution"] = geo_distribution

# Booking Lead time
lead_time_distribution = df["lead_time"].describe().to_dict()
analytics_cache["lead_time_distribution"] = lead_time_distribution

# ✅ Vector Embedding for Q&A using FAISS
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
vector_data = df.astype(str).agg(" ".join, axis=1).tolist()
embeddings = model.encode(vector_data, convert_to_tensor=True)

# Store in FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.cpu().numpy())  # Ensure it's on CPU

# ✅ Define API Models
class QueryModel(BaseModel):
    query: str

# 🔹 API Endpoints

@app.get("/")
def home():
    return {"message": "Welcome to the Hotel Booking Analytics API!"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running fine"}

@app.post("/analytics")
def get_analytics():
    """Returns precomputed analytics."""
    return analytics_cache

@app.post("/ask")
def ask_question(query_data: QueryModel):
    """Handles Natural Language Q&A using FAISS + LLM"""
    query_embedding = model.encode([query_data.query], convert_to_tensor=True).cpu().numpy()
    _, result_indices = index.search(query_embedding, k=1)  # Retrieve best match
    matched_row = df.iloc[result_indices[0][0]].to_dict()
    return {"question": query_data.query, "answer": matched_row}

# Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)
