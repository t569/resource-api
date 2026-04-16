# 🌌 Digital Garden Vector API

A serverless Retrieval-Augmented Generation (RAG) backend and GitOps ingestion pipeline built to power a static Digital Garden. 

This API decouples heavy vector operations from the static frontend, utilizing FastAPI to orchestrate OpenAI embeddings, Pinecone vector storage, and autonomous GitHub Pull Requests for human-in-the-loop review.

## 🏗 Architecture

The system is designed around a strictly decoupled "GitOps" philosophy:
1. **The Staging Phase:** Submissions are intercepted by the API, parsed by an LLM to generate deep-dive technical queries, and committed to a new branch on the frontend repository as a Markdown file. 
2. **The Review Phase:** The API opens a Pull Request. A human reviews the generated output and merges it into the static site.
3. **The Injection Phase:** Once merged, the approved payload is sent back to the API to be mathematically embedded and permanently injected into the Vector Database.

## ✨ Core Features
* **Semantic Search:** Uses OpenAI's `text-embedding-3-small` to convert search queries into vectors and find nearest neighbors in 1536-dimensional space.
* **LLM Augmentation:** Automatically analyzes submitted resources and generates specific, advanced Google search queries for further technical exploration.
* **Autonomous GitOps:** Utilizes `PyGithub` to programmatically branch, write, and open Pull Requests directly against the frontend repository.
* **Vault Security:** Write-endpoints are protected by a custom header-based API key (`X-Garden-Key`) to prevent database corruption from unauthorized requests.

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
* Python 3.12+
* A [Pinecone](https://app.pinecone.io/) account with a 1536-dimension index using the `cosine` metric.
* An [OpenAI](https://platform.openai.com/) API Key.
* A GitHub Personal Access Token (Classic) with `repo` scope.

### 2. Environment Setup
Clone the repository and spin up a virtual environment:

```bash
git clone [https://github.com/t569/resource-api.git](https://github.com/t569/resource-api.git)
cd resource-api
python -m venv venv

# Activate the environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

3. The Vault (.env)
Create a .env file in the root directory and populate it with your keys. Never commit this file.

Plaintext
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
GITHUB_TOKEN=your_github_personal_access_token
GARDEN_API_KEY=your_custom_secure_password # Used to lock the POST endpoints
4. Boot the Server
Run the Uvicorn ASGI server:

Bash
uvicorn main:app --reload
The API will be available at http://127.0.0.1:8000. You can access the interactive Swagger UI at http://127.0.0.1:8000/docs.

📡 API Reference
GET /resources/search
(Publicly Accessible)
Accepts a natural language query, converts it to a vector, and returns the top 5 semantically similar resources from the Pinecone database.

Parameters: q (string), top_k (integer, default=5)

POST /resources/stage
(Protected: Requires X-Garden-Key header)
Analyzes a resource, generates Markdown, and opens a Pull Request on the target GitHub repository. Does not modify the vector database.

Body:

JSON
{
  "title": "Resource Title",
  "url": "[https://example.com](https://example.com)",
  "description": "Brief overview...",
  "category": "Architecture",
  "tags": ["system-design", "backend"]
}
POST /resources/inject
(Protected: Requires X-Garden-Key header)
Takes the finalized, approved data from the staging phase, embeds it via OpenAI, and upserts the vector and metadata into Pinecone.

☁️ Deployment (Render)
This API is configured for seamless deployment on containerized cloud hosts like Render.

Connect your GitHub repository to Render as a Web Service.

Select Python 3 as the environment.

Set the Build Command: pip install -r requirements.txt

Set the Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

Inject the 4 environment variables listed in the .env setup.

Force the correct Python version by adding an environment variable: PYTHON_VERSION = 3.12.0.