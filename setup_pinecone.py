import os
from dotenv import load_dotenv
import pinecone
from pinecone import ServerlessSpec

# Load environment variables
load_dotenv()

# Retrieve API Key and Environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ NEW Pinecone Initialization
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# List existing indexes
existing_indexes = pinecone_client.list_indexes()
print("Available Pinecone Indexes:", existing_indexes)

# Define an index name
index_name = "factchecker"

# ✅ Only create the index if it doesn’t already exist
if index_name not in [index["name"] for index in existing_indexes]:
    pinecone_client.create_index(
        name=index_name,
        spec=ServerlessSpec(
            cloud="aws",  # ✅ Pinecone is on AWS
            region="us-east-1"  # ✅ Your Pinecone region
        ),
        dimension=1536  # ✅ Required for dense vector indexes (1536 for OpenAI embeddings)
    )
    print(f"Pinecone index '{index_name}' created successfully!")
else:
    print(f"Pinecone index '{index_name}' already exists. Skipping creation.")

pinecone_client.delete_index(index_name)
print(f"Deleted index '{index_name}'. Now re-creating it...")

pinecone_client.create_index(
    name=index_name,
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ),
    dimension=1536
)

print(f"Pinecone index '{index_name}' re-created!")
