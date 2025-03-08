# AI Fact Checker

## Overview
AI Fact Checker is an intelligent fact-verification tool designed to analyze claims using a multi-agent system. It leverages **Guardrails AI** for structured output validation, **LangChain** for retrieval and reasoning, and **Gradio** for an interactive user interface.

## Features
- **Automated Fact Verification**: Extracts facts and validates them against reliable sources.
- **Bias & Misinformation Detection**: Identifies bias levels and misinformation risks in claims.
- **Multi-Agent Reasoning**: Uses LangChain and Guardrails AI to ensure structured and validated responses.
- **User-Friendly Interface**: Built with **Gradio** for seamless user interactions.

## Tech Stack
- **Backend**: Python, LangChain, Pinecone (Vector DB), Guardrails AI
- **Frontend**: Gradio
- **Deployment**: Local execution (can be extended to Docker or Hugging Face Spaces)

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AI-FactChecker.git
cd AI-FactChecker
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv factchecker-env
source factchecker-env/bin/activate  # On Windows: factchecker-env\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.env` file in the project root and add:
```ini
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_env
```

### 4. Run the Application
```bash
python app.py
```

This will launch the Gradio interface for user interaction.

## Usage
1. Enter a claim in the input box.
2. The system retrieves relevant information and analyzes the claim.
3. The results display the **retrieved fact**, **bias score**, **verification**, and **final verdict**.



