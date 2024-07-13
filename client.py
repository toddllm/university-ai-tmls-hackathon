import requests

class OpenAIClient:
    def __init__(self, embeddings_base_url, ask_base_url, api_key):
        self.embeddings_base_url = embeddings_base_url
        self.ask_base_url = ask_base_url
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def create_embedding(self, model, input_text, encoding_format="float"):
        url = f"{self.embeddings_base_url}/v1/embeddings"
        payload = {
            "model": model,
            "input": input_text,
            "encoding_format": encoding_format
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def ask_question(self, query):
        url = f"{self.ask_base_url}/ask"
        payload = {
            "query": query
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

# Initialize the client
client = OpenAIClient(embeddings_base_url="http://127.0.0.1:8001", ask_base_url="http://127.0.0.1:8000", api_key="lit")

# Create an embedding
embedding_response = client.create_embedding(
    model="jina-embeddings-v2-small-en",
    input_text="The food was delicious and the waiter...",
    encoding_format="float"
)
print("Embedding Response:", embedding_response)

# Ask a question
question_response = client.ask_question(
    query="What are the application deadlines for the big data analytics MSc for 2024?"
)
print("Question Response:", question_response)
