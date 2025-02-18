from fastapi import FastAPI, HTTPException, Query
from huggingface_hub import InferenceClient
import os
import logging
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("HUGGING_FACE_API_TOKEN")

# Initialize client
client = InferenceClient(api_key=api_key)
model = "meta-llama/Llama-3.2-3B-Instruct"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CollabAI Chat API", version="1.0")

@app.get("/")
async def home():
    return "Chat API is running!"

@app.head("/health")
async def home():
    return "Healthy"

@app.post("/chat/")
async def chat_with_model(query: str = Query(..., description="User input message")):
    """
    API endpoint to interact with a Hugging Face chat model.

    - Retries up to 3 times if an error occurs before failing.
    - Prioritizes smaller models when the selected model fails.
    """

    messages = [{"role": "system", "content": "You are CollabAI, a knowledgeable and efficient AI assistant. Respond concisely and helpfully to user queries without unnecessary introductions."},  
            {"role": "user", "content": query}]

    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=0.7,
                stream=False
            )

            return {
                "response": response.choices[0].message.content,
                "model_used": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
            if attempt < max_retries:
                await asyncio.sleep(1)  # Short delay before retrying

    raise HTTPException(status_code=503, detail="The model is currently unavailable after multiple retries. Please try again later.")
