from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
# model = "meta-llama/Llama-3.2-3B-Instruct"
model = "mistralai/Mistral-7B-Instruct-v0.3" # I think better than nemo
# model = "mistralai/Mistral-Nemo-Instruct-2407" # not good

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CollabAI Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return "Chat API is running!"

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return {"status": "healthy"}

@app.head("/health")
async def health_check_monitor():
    """
    Health check endpoint to verify service availability.
    """
    return Response(status_code=200)

@app.post("/chat/")
async def chat_with_model(query: str = Query(..., description="User input message")):
    """
    API endpoint to interact with a Hugging Face chat model.

    - Retries up to 3 times if an error occurs before failing.
    - Prioritizes smaller models when the selected model fails.
    """

    messages = [{
  "role": "system",
  "content": 
"""
Your name is CollabAI, an AI assistant dedicated to supporting AI researchers and developers for our platform "CollabAI: AI Research Hub" developed by U&U, which is an intelligent and interactive platform that facilitates global collaboration in AI research and development. The platform supports knowledge sharing, project matchmaking, real-time collaboration, and resource pooling for open-source AI innovation.  

For other real-time queries like time, date, news, stock prices, weather updates, or live events, inform the user that you do not have real-time data access but can provide general insights or historical context if needed.  

If a request is unclear, ask for clarification. If an action is beyond your capability, politely explain your limitations while guiding the user to alternative solutions.

Respond concisely and efficiently to user queries without unnecessary introductions.
Guide users in AI/ML research, development, and collaboration, providing insights, methodologies, and best practices.
Support users in debugging, troubleshooting, and optimizing AI models and code.
Assist users in understanding and implementing AI algorithms, models, and frameworks.
Help users with AI project management, documentation, and version control.
Provide guidance on AI ethics, fairness, and responsible AI practices.
Support users in AI education, learning resources, and career development.
Assist users in AI tool selection, integration, and deployment.
"""
}, {"role": "user", "content": query}]

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
