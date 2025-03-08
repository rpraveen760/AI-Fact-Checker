import os
import pinecone
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from typing import TypedDict
from guardrails import Guard

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "factchecker"
index = pinecone_client.Index(index_name)

# 🔍 Step 1: Search Agent - Retrieve the Best Fact from Pinecone
def search_agent(state):
    user_claim = state["claim"]
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=user_claim
    ).data[0].embedding

    # Query Pinecone
    search_results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    if search_results["matches"]:
        best_fact = search_results["matches"][0]["metadata"]["text"]
    else:
        best_fact = "No relevant fact found."

    return {"claim": user_claim, "retrieved_fact": best_fact}

# 🏳️ Step 2: Bias Detection Agent
def bias_agent(state):
    claim = state["claim"]
    system_prompt = "Analyze the following claim for biased, emotional, or misleading language. Return only 'Biased' or 'Neutral'."

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=claim)])

    return {"bias": response.content}

# ✅ Step 3: Fact-Checking Agent
def fact_checker_agent(state):
    claim = state["claim"]
    retrieved_fact = state["retrieved_fact"]

    system_prompt = f"""Compare the following claim to the retrieved fact-checking statement. 
    Determine if the claim is: 
    - 'Supported' (matches verified fact-checking sources)
    - 'Contradicted' (directly opposed by fact-checking sources)
    - 'Unverified' (not enough information to verify)
    Only return one of these three words."""



    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=f"Claim: {claim}\nFact: {retrieved_fact}")])

    return {"verification": response.content}

# 🏆 Step 4: Verdict Agent
import json
from guardrails import Guard

def verdict_agent(state):
    verification = state["verification"]
    bias = state["bias"]
    guard = Guard.from_rail("factcheck.rail")

    # Example: We'll hard-code actual_truth for demonstration
    # In reality, you might fetch from a known database or agent logic
    #actual_truth_str = "Scientists worldwide agree Earth is round."

    raw_response = {
        "fact_check_response": {
            #"actual_truth": actual_truth_str,
            "verified_fact": state["retrieved_fact"],
            "bias_score": bias,
            "misinformation_risk": "Medium" if verification == "Contradicted" else "Low",
            "final_verdict": "True" if (verification == "Supported" and bias == "Neutral") else (
                "Misleading" if bias == "Biased" else "Fake"
            )
        }
    }

    raw_response_str = json.dumps(raw_response)
    parsed_output = guard.parse(raw_response_str)

    print("VALIDATION PASSED?", parsed_output.validation_passed)
    print("VALIDATED OUTPUT:", parsed_output.validated_output)
    print("REASK?", parsed_output.reask)

    # Fallback approach:
    validated_dict = parsed_output.validated_output.get("fact_check_response", {})

    # Provide safe defaults if fields are missing
    final_verdict = validated_dict.get("final_verdict", None)
    actual_truth = validated_dict.get("actual_truth", "No actual truth found.")

    if parsed_output.validation_passed and parsed_output.validated_output:
        return {
            "retrieved_fact": state["retrieved_fact"],
            "bias": bias,
            "verification": verification,
            "final_verdict": final_verdict,
           # "actual_truth": actual_truth
        }
    else:
        return {
            "retrieved_fact": state["retrieved_fact"],
            "bias": bias,
            "verification": verification,
            "final_verdict": None,
            "actual_truth": None
        }


# 🏗️ Step 5: Build the LangGraph Workflow
class FactCheckState(TypedDict):
    claim: str
    retrieved_fact: str
    bias: str
    verification: str
    final_verdict: str
    #actual_truth: str

workflow = StateGraph(FactCheckState)

workflow.add_node("Search", search_agent)
workflow.add_node("BiasAnalysis", bias_agent)
workflow.add_node("FactCheck", fact_checker_agent)
workflow.add_node("Verdict", verdict_agent)

# Define edges (execution order)
workflow.add_edge("Search", "BiasAnalysis")
workflow.add_edge("Search", "FactCheck")
workflow.add_edge("BiasAnalysis", "Verdict")
workflow.add_edge("FactCheck", "Verdict")

workflow.set_entry_point("Search")
#workflow.set_exit_point("Verdict")

# 🚀 Step 6: Run the Multi-Agent System
def run_fact_checker(claim: str) -> dict:
   # """Run the multiagent pipeline for a single claim and return the result dict."""
    executor = workflow.compile()
    result = executor.invoke({"claim": claim})
    return result

#
# if __name__ == "__main__":
#     user_claim = input("Enter a claim to fact-check: ")
#     # Compile the workflow
#     executor = workflow.compile()
#
#     # Run the workflow
#     result = executor.invoke({"claim": user_claim})
#
#     print(f"\n🔍 Claim: {user_claim}")
#     print(f"📌 Retrieved Fact: {result['retrieved_fact']}")
#     print(f"🏳️ Bias Analysis: {result['bias']}")
#     print(f"✅ Fact Verification: {result['verification']}")
#     print(f"🏆 Final Verdict: {result['final_verdict']}")
