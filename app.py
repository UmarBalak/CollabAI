from fastapi import FastAPI, HTTPException, Query
from huggingface_hub import InferenceClient
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("HUGGING_FACE_API_TOKEN")

# Initialize client
client = InferenceClient(api_key=api_key)

# Load models from environment
models = os.getenv("MODELS").split(",")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CollabAI Chat API", version="1.0")

@app.get("/")
async def home():
    return "Chat API is running!"

@app.post("/chat/")
async def chat_with_model(
    user_message: str = Query(..., description="User input message"),
    preferred_model: str = Query(None, description="Preferred model name")
):
    """
    API endpoint to interact with a Hugging Face chat model.
    
    - If the preferred model is unavailable, it falls back to another model.
    - Prioritizes smaller models when the selected model fails.
    """
    # Determine model priority
    model_queue = models[:]  # Copy model list
    if preferred_model and preferred_model in models:
        model_queue.remove(preferred_model)
        model_queue.insert(0, preferred_model)  # Move user choice to top

    messages = [{"role": "user", "content": user_message}]
    
    for model in model_queue:
        try:
            response = client.chat.completions.create(
                model=model.strip(),
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
            logger.warning(f"Model {model} failed: {str(e)}")
    
    raise HTTPException(status_code=503, detail="All models are currently unavailable. Please try again later.")
