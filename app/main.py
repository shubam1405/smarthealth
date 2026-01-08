from fastapi import FastAPI

app = FastAPI(
    title="Smart Healthcare Diagnostic Platform",
    description="AI-powered healthcare risk and diagnostic system",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "Backend is running successfully"}
