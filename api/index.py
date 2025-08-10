from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mangum import Mangum  # to adapt FastAPI to serverless
from main import app

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Vercel!"}

# This is important for Vercel
handler = Mangum(app)
