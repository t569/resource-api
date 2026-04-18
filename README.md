# Digital Garden RAG API

A serverless Retrieval-Augmented Generation (RAG) pipeline and vector search engine for building a dynamic Digital Garden. 

This API ingests technical resources (links, descriptions, and tags), embeds them into a vector database, and provides a semantic search endpoint. When a search is performed, it retrieves the most relevant niche resources and uses an LLM to generate highly contextual, advanced follow-up search queries to fuel deeper exploration.

## 🏗 Architecture & Tech Stack

* **Compute/Hosting:** [Vercel Serverless Functions](https://vercel.com/) (Hobby Tier - Free)
* **Backend Framework:** [FastAPI](https://fastapi.tiangolo.com/) + Python
* **Vector Database:** [Pinecone Serverless](https://pinecone.io/) (Free Tier)
* **Embeddings:** `llama-text-embed-v2` (Via Pinecone Native Inference - Free)
* **Generative AI (LLM):** [Groq](https://groq.com/) using Llama 3 (Free Tier)
* **CI/CD:** Automated deployments via GitHub integration

---

## 🚀 Local Development Setup

To test and develop the API on your local machine before pushing to production:

### 1. Prerequisites

* Python 3.10+ installed
* [Vercel CLI](https://vercel.com/docs/cli) installed (`npm i -g vercel`)

### 2. Installation

Clone the repository and set up your Python virtual environment:

```bash
git clone [https://github.com/t569/resource-api.git](https://github.com/t569/resource-api.git)
cd resource-api
python -m venv venv

# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Vercel CLI Authentication & Linking

Authenticate your local machine with Vercel and link it to your project:
```bash

vercel login
# Follow the prompt to log in via GitHub

vercel link
# Follow the prompts to link to your existing `resource-api` project
```

### 4. Sync Environment Variables

Instead of creating a `.env` file manually, pull your secure keys directly from your Vercel project:

```bash
vercel env pull .env
```

This will securely download `RESOURCE_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `GROQ_API_KEY` to your local machine.

### 5. Run the Local Development Server

Use Vercel's local simulation to accurately mimic the serverless cloud environment:

```bash
vercel dev
```
Your API will now be running locally (usually at `http://localhost:3000`).

* Interactive Docs: Go to `http://localhost:3000/docs` to test your endpoints via Swagger UI.

---

## 🌐 Deployment (CI/CD)

This project is connected to GitHub for Continuous Deployment.

To deploy changes to production, simply commit and push to the `main` branch:

```bash
git add .
git commit -m "Your update message"
git push origin main
```
Vercel will automatically detect the push, rebuild the Python environment, and deploy the updated API to your live `.vercel.app` URL.

---

## 🔑 Environment Variables
If you need to rotate keys or set up the project from scratch, these are the required environment variables in the Vercel Dashboard:

* `RESOURCE_API_KEY`: Your custom secret password used to protect the injection endpoint.

* `PINECONE_API_KEY`: Your API key from the Pinecone dashboard.

* `PINECONE_INDEX_NAME`: The name of your serverless index (e.g., `digital-garden`). Must have a dimension of `1024`.

* `GROQ_API_KEY`: Your API key from the Groq console for generating AI search suggestions.

---

## 📡 API Endpoints
1. Inject a Resource (Protected)
`POST /resources/inject`

Embeds a new resource and saves it to the vector database.

**Headers**: `X-Garden-Key: <YOUR_RESOURCE_API_KEY>`

**Payload**:

```json
{
    "title": "Understanding gVisor Sandboxes",
    "url": "[https://gvisor.dev/docs/](https://gvisor.dev/docs/)",
    "description": "Deep dive into application kernels and secure container isolation.",
    "category": "Architecture",
    "tags": ["docker", "gvisor", "security"]
}
```

2. Semantic Search & RAG Generation (Public)
`GET /search?q={query}&top_k={number}`

Retrieves the closest matching resources and generates tailored follow-up queries using Groq.

**Example**: `/search?q=bare+metal+programming&top_k=5`

**Response**:

```json
{
    "results": [
        {
            "id": "uuid-string",
            "title": "...",
            "url": "...",
            "description": "...",
            "tags": ["..."]
        }
    ],
    "ai_suggested_queries": [
        "Advanced cooperative multitasking patterns in bare metal ARM",
        "Interrupt vector table configuration for Cortex-M0+",
        "Implementing zero-allocation memory managers in C"
    ]
}
```


