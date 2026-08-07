from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predict import generate_predictions
import numpy as np

app = FastAPI(title="SmartStock AI API")

# allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# health check
@app.get("/")
def home():
    return {
        "status": "SmartStock AI running"
    }

@app.get("/predictions")
def get_predictions():
    df = generate_predictions()

    # convert NaN → None (JSON-safe)
    df = df.replace({np.nan: None})

    records = df.to_dict(orient="records")

    # ensure numpy types are converted
    def clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    return [
        {k: clean(v) for k, v in row.items()}
        for row in records
    ]