import os
import json
import uuid
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from pinecone import Pinecone
from openai import OpenAI
from github import Github
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Digital Garden API", description="Serverless Vector Search & Autocommit API")

# --- CORS Middleware (Crucial for Cloud Deployment) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",           # Local Quartz
        "https://t569.github.io"           # Live Quartz
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize External Clients ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("digital-garden-resources") # Ensure this matches your Pinecone index name

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize GitHub Client
# You will need to generate a Personal Access Token (Classic) in GitHub with 'repo' scope
g = Github(os.environ.get("GITHUB_TOKEN"))
BLOG_REPO_NAME = "t569/blog" # Your target repository

# --- Data Models ---
class ResourceBase(BaseModel):
    title: str
    url: HttpUrl
    description: Optional[str] = ""
    category: str
    subcategory: Optional[str] = ""
    tags: List[str] = []

class Resource(ResourceBase):
    id: str

# --- Helper Functions ---
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding

# --- Endpoints ---
@app.post("/resources/", status_code=201)
def add_resource(resource: ResourceBase):
    """Embeds the resource, generates deep-dive queries, upserts to Pinecone, and commits to GitHub."""
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

    # 2. Generate Vector Embeddings
    text_to_embed = f"Title: {resource.title}. Description: {resource.description}. Tags: {', '.join(resource.tags)}."
    vector = get_embedding(text_to_embed)
    
    # 3. Upsert to Pinecone
    metadata = {
        "title": resource.title,
        "url": str(resource.url),
        "description": resource.description or "",
        "category": resource.category,
        "subcategory": resource.subcategory or "",
        "tags": resource.tags,
        "google_queries": google_queries 
    }
    index.upsert(vectors=[(resource_id, vector, metadata)])
    
    # 4. Generate the Quartz Markdown File
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

    # 5. Push to GitHub Automatically
    try:
        repo = g.get_repo(BLOG_REPO_NAME)
        file_path = f"content/resources/{safe_filename}"
        repo.create_file(
            path=file_path,
            message=f"API Autocommit: Added resource {resource.title}",
            content=markdown_content,
            branch="main"
        )
    except Exception as e:
        print(f"GitHub push failed: {e}. Resource is in Pinecone but not in Quartz.")
        # We don't fail the API call if GitHub fails, but we log it.

    return {"id": resource_id, "message": "Resource embedded and committed to GitHub.", "queries": google_queries}

@app.get("/resources/search")
def search_resources(q: str = Query(..., description="Semantic search query"), top_k: int = Query(5)):
    """Converts the query to a vector and finds nearest neighbors."""
    query_vector = get_embedding(q)
    
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
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