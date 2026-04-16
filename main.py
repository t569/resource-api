import os
import json
import uuid
from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from pinecone import Pinecone
from openai import OpenAI
from github import Github
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Digital Garden API", description="Serverless Vector Search & GitOps Ingestion")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "https://t569.github.io/blog"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security Wrapper ---
# Define the name of the header the frontend must send
API_KEY_NAME = "X-Garden-Key"
# Read your custom invented password from the environment
SECRET_KEY = os.environ.get("GARDEN_API_KEY") 
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == SECRET_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")

# --- Initialize External Clients ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("resource-api-garden") 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"

g = Github(os.environ.get("GITHUB_TOKEN"))
BLOG_REPO_NAME = "t569/blog"

# --- Data Models ---
class ResourceSubmission(BaseModel):
    title: str
    url: HttpUrl
    description: Optional[str] = ""
    category: str
    tags: List[str] = []

class ConfirmedResource(ResourceSubmission):
    id: str
    google_queries: List[str] = []

# --- Helper Functions ---
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding

# --- Endpoints ---

# takes the raw link, hits the LLM to generate the deep-dive queries,
# generate the quartz markdown file and open a github PR
@app.post("/resources/stage", status_code=201, dependencies=[Depends(verify_api_key)])
def stage_resource(resource: ResourceSubmission):
    """Hits the LLM, writes the Quartz Markdown, and opens a PR. Does NOT inject to Pinecone."""
    resource_id = str(uuid.uuid4())
    
    # 1. Generate Deep-Dive Google Queries
    prompt = f"""
    Analyze this technical resource: Title: {resource.title}, Description: {resource.description}, Tags: {', '.join(resource.tags)}
    Generate exactly 3 highly specific, advanced Google search queries a systems engineer would use to deeply explore the underlying concepts of this resource. Return ONLY a valid JSON object with a single key "queries" containing an array of 3 strings.
    """
    
    llm_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    google_queries = json.loads(llm_response.choices[0].message.content).get("queries", [])

    # 2. Generate the Quartz Markdown File
    safe_filename = "".join([c if c.isalnum() else "-" for c in resource.title]).lower() + ".md"
    markdown_content = f"""---
title: "{resource.title}"
tags: {resource.tags}
---

# {resource.title}
**Link:** [{str(resource.url)}]({str(resource.url)})

### Overview
{resource.description}

### Deep Dive Queries
"""
    for query in google_queries:
        formatted_query = query.replace(" ", "+")
        markdown_content += f"* 🔍 [{query}](https://www.google.com/search?q={formatted_query})\n"
        
    markdown_content += "\n---\n**System Architecture Nodes:** [[resources/index|Compendium Hub]] | [[about|About the Architect]]\n"

    # 3. Push to a Review Branch & Open PR
    try:
        repo = g.get_repo(BLOG_REPO_NAME)
        branch_name = f"review/{resource_id[:8]}" 
        source = repo.get_branch("main")
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=source.commit.sha)
        
        repo.create_file(
            path=f"content/resources/{safe_filename}",
            message=f"API Staging: {resource.title}",
            content=markdown_content,
            branch=branch_name
        )
        
        # Open a Pull Request
        pr_body = f"""
        ### Resource Review
        Please review this staging submission. If approved, merge this PR to update the Quartz graph.
        
        **To inject into Pinecone, fire the `/resources/inject` endpoint with this payload:**
        ```json
        {{
            "id": "{resource_id}",
            "title": "{resource.title}",
            "url": "{str(resource.url)}",
            "description": "{resource.description}",
            "category": "{resource.category}",
            "tags": {json.dumps(resource.tags)},
            "google_queries": {json.dumps(google_queries)}
        }}
        ```
        """
        repo.create_pull(
            title=f"Resource Staging: {resource.title}",
            body=pr_body,
            head=branch_name,
            base="main"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GitHub PR failed: {e}")

    return {"message": "Resource staged. Pull Request created.", "id": resource_id, "queries": google_queries}

# once we review and merge the PR of our markdown into our blog we hit this endpoint to inject it into pinecone
@app.post("/resources/inject", status_code=201, dependencies=[Depends(verify_api_key)])
def inject_resource(resource: ConfirmedResource):
    """Embeds the approved payload and pushes it to Pinecone."""
    # 1. Generate Vector Embeddings
    text_to_embed = f"Title: {resource.title}. Description: {resource.description}. Tags: {', '.join(resource.tags)}."
    vector = get_embedding(text_to_embed)
    
    # 2. Upsert to Pinecone
    metadata = {
        "title": resource.title,
        "url": str(resource.url),
        "description": resource.description or "",
        "category": resource.category,
        "tags": resource.tags,
        "google_queries": resource.google_queries 
    }
    index.upsert(vectors=[(resource.id, vector, metadata)])
    
    return {"message": "Resource successfully injected into Pinecone."}


@app.get("/resources/search")
def search_resources(q: str = Query(..., description="Semantic search query"), top_k: int = Query(5)):
    """Public search endpoint - No API key required."""
    query_vector = get_embedding(q)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    formatted_results = []
    for match in results.matches:
        meta = match.metadata
        formatted_results.append({
            "id": match.id,
            "title": meta["title"],
            "url": meta["url"],
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "google_queries": meta.get("google_queries", [])
        })
        
    return formatted_results