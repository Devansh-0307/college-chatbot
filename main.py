import os
import requests
import difflib
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

# Load environment variables (Gemini key)
load_dotenv()
genai.configure(api_key=os.getenv("Gemini_api_key"))

# Load QA dataset
df = pd.read_csv("college_qa.csv")
college_qa = dict(zip(df['question'].str.lower(), df['answer']))

# Create FastAPI app
app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 templates (for rendering index.html)
templates = Jinja2Templates(directory="templates")

# Route: Serve index.html at /
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ask Gemini model
def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini failed:", e)
        return None

# Ask Ollama model (e.g. mistral)
def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",  # You can use llama2 or phi
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print("Ollama failed:", e)
        return "Sorry, I'm unable to process your request right now."

# API route for chatbot
@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question_text = data.get("question", "").lower()

    try:
        print("Received question:", question_text)

        # 1. Try CSV matching
        match = difflib.get_close_matches(question_text, college_qa.keys(), n=1, cutoff=0.85)
        if match:
            return {"answer": college_qa[match[0]]}

        # 2. Try Gemini
        print("Trying Gemini...")
        gemini_answer = ask_gemini(question_text)
        if gemini_answer:
            return {"answer": gemini_answer}

        # 3. Fallback to Ollama
        print("Gemini failed. Trying Ollama...")
        ollama_answer = ask_ollama(question_text)
        return {"answer": ollama_answer}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {str(e)}"}
