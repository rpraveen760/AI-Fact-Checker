import os
import pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Connect to the Pinecone index
index_name = "factchecker"
index = pinecone_client.Index(index_name)

# Sample fact-checking data
documents = [
    "NASA confirms that the Earth is not flat.",
    "Vaccines are effective in preventing diseases.",
    "5G technology does not cause COVID-19.",
    "Climate change is real and caused by human activity.",
]

# Insert documents into Pinecone
for i, doc in enumerate(documents):
    # Generate embeddings using OpenAI
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=doc
    ).data[0].embedding

    # Store in Pinecone
    index.upsert(
        vectors=[(f"doc-{i}", embedding, {"text": doc})],
        namespace=""  # Ensures it's stored in the default namespace
    )

print("✅ Data inserted into Pinecone successfully!")
