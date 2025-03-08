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

# Function to retrieve the most relevant fact-checking statement
def retrieve_fact(user_query):
    # Convert query to embedding
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=user_query
    ).data[0].embedding

    # Search in Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=1,  # Get the most relevant result
        include_metadata=True  # ✅ Now we retrieve metadata!
    )

    # Extract the best match
    if search_results["matches"]:
        best_match = search_results["matches"][0]

        # Retrieve the actual text from metadata (assuming we store metadata later)
        fact_text = best_match.get("metadata", {}).get("text", "No text found.")

        print(f"🔍 Best Match ID: {best_match['id']}")
        return f"✅ Closest Fact-Check: {fact_text}"


# Test with a user query
user_input = input("Enter a claim to fact-check: ")
result = retrieve_fact(user_input)
print(result)
