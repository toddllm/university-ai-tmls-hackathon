import nest_asyncio
import os
import logging
from typing import List, Literal, Union
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import openai
import litserve as ls

# Initialize FastAPI
app = FastAPI()

# Allows nested access to the event loop
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a request model for the ask endpoint
class Question(BaseModel):
    query: str

# Define allowed embedding models using Literal and dictionary keys
MODEL_MAPPING = {
    "jina-embeddings-v2-small-en": "jinaai/jina-embeddings-v2-small-en",
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1",
}

# Define embedding request model
EMBEDDING_MODELS = Literal[tuple(MODEL_MAPPING.keys())]

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: EMBEDDING_MODELS
    encoding_format: Literal["float"]

# Define embedding response models
class Embedding(BaseModel):
    embedding: List[float]
    index: int
    object: Literal["embedding"] = "embedding"

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    data: List[Embedding]
    model: EMBEDDING_MODELS
    object: Literal["list"] = "list"
    usage: Usage

BERT_CLASSES = ["NomicBertModel", "BertModel"]

class EmbeddingAPI(ls.LitAPI):
    def setup(self, device, model_id="jina-embeddings-v2-small-en"):
        """Setup the model and tokenizer."""
        logging.info(f"Loading model: {model_id}")
        self.model_id = model_id
        self.model_name = MODEL_MAPPING[model_id]
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def decode_request(self, request: EmbeddingRequest, context) -> List[str]:
        """Decode the incoming request and prepare it for prediction."""
        context["model"] = request.model

        # load model if different from the active model
        if request.model != self.model_id:
            self.setup(self.device, request.model)

        sentences = [request.input] if isinstance(request.input, str) else request.input
        context["total_tokens"] = sum(
            len(self.tokenizer.encode(text)) for text in sentences
        )
        return sentences

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def predict(self, x) -> List[List[float]]:
        is_bert_instance = self.model.__class__.__name__ in BERT_CLASSES
        if is_bert_instance:
            encoded_input = self.tokenizer(
                x, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                return (
                    self.mean_pooling(model_output, encoded_input["attention_mask"])
                    .cpu()
                    .numpy()
                )

        return self.model.encode(x)

    def encode_response(self, output, context) -> EmbeddingResponse:
        """Encode the embedding output into the response model."""
        embeddings = [
            Embedding(embedding=embedding.tolist(), index=i)
            for i, embedding in enumerate(output)
        ]
        return EmbeddingResponse(
            data=embeddings,
            model=context["model"],
            usage=Usage(
                prompt_tokens=context["total_tokens"],
                total_tokens=context["total_tokens"],
            ),
        )

@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )

@app.post("/ask")
async def ask_question(question: Question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question.query}],
            max_tokens=150
        )
        return {"answer": response.choices[0].message['content'].strip()}
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize the embedding server
api = EmbeddingAPI()
server = ls.LitServer(api, accelerator="auto", api_path="/v1/embeddings")

if __name__ == "__main__":
    import multiprocessing
    import uvicorn

    def start_embedding_server():
        server.run(port=8001)

    def start_fastapi_server():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    # Start both servers
    p1 = multiprocessing.Process(target=start_embedding_server)
    p2 = multiprocessing.Process(target=start_fastapi_server)
    
    p1.start()
    p2.start()

    p1.join()
    p2.join()
