import os
import json
import uuid
from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv

# API Endpoints: https://resource-api-eight.vercel.app/docs
# https://resource-api-eight.vercel.app

# Load environment variables for local development
load_dotenv()

app = FastAPI(
    title="Digital Garden RAG API", 
    description="Vector Search Pipeline with LLM-Augmented Follow-up Queries"
)

# --- Middleware & Security ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update this to your specific frontend/blog domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_NAME = "X-Garden-Key"
SECRET_KEY = os.environ.get("RESOURCE_API_KEY") 
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == SECRET_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Unauthorized")

# --- External Clients ---
# Pinecone handles Vector Storage AND the Llama-Text-Embed-v2 inference
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "resource-api-garden")) 
EMBEDDING_MODEL = "llama-text-embed-v2"

# Groq handles the fast, free LLM text generation for the dynamic search suggestions
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Data Models ---
class ResourceSubmission(BaseModel):
    title: str
    url: HttpUrl
    description: Optional[str] = ""
    category: str
    tags: List[str] = []

class ResourceResponse(BaseModel):
    id: str
    title: str
    url: str
    description: str
    tags: List[str]

class SearchResponse(BaseModel):
    results: List[ResourceResponse]
    ai_suggested_queries: List[str]

# --- Helper Functions ---
def get_embedding(text: str, is_query: bool = False) -> List[float]:
    """Generates a vector embedding using Pinecone's Native Inference."""
    input_type = "query" if is_query else "passage"
    response = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[text],
        parameters={"input_type": input_type, "truncate": "END"}
    )
    return response[0].values

def generate_smart_queries(search_term: str, retrieved_context: str) -> List[str]:
    """Passes the retrieved DB context to an LLM to generate tailored follow-up queries."""
    prompt = f"""
    The user searched for: "{search_term}".
    Based on our database, we found these relevant resources:
    {retrieved_context}
    
    Based on what they searched and what we actually have in our database, generate exactly 3 highly specific, 
    advanced search queries they could use next to explore deeper into these specific niches. 
    Return ONLY a raw JSON array of 3 strings. No markdown formatting, no explanations.
    """
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", # Fast, free model
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        response_text = chat_completion.choices[0].message.content
        # Ensure we parse out the array safely
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            # Groq sometimes wraps arrays in a dict key if prompted for a json_object
            return next(iter(parsed.values()))
        return parsed
    except Exception as e:
        print(f"LLM Generation failed: {e}")
        return ["Advanced " + search_term + " architecture", search_term + " edge cases"]

# --- Endpoints ---

@app.post("/resources/inject", status_code=201, dependencies=[Depends(verify_api_key)])
def inject_resource(resource: ResourceSubmission):
    """Embeds the resource using Llama-Embed and pushes it to Pinecone."""
    resource_id = str(uuid.uuid4())
    text_to_embed = f"Title: {resource.title}. Description: {resource.description}. Tags: {', '.join(resource.tags)}."
    
    vector = get_embedding(text_to_embed, is_query=False)
    
    metadata = {
        "title": resource.title,
        "url": str(resource.url),
        "description": resource.description or "",
        "category": resource.category,
        "tags": resource.tags
    }
    index.upsert(vectors=[(resource_id, vector, metadata)])
    
    return {"message": "Resource successfully injected.", "id": resource_id}

@app.get("/search", response_model=SearchResponse)
def search_pipeline(q: str = Query(..., description="User search query"), top_k: int = Query(5)):
    """
    The core RAG pipeline:
    1. Embeds the user query.
    2. Retrieves top niche resources from Pinecone.
    3. Feeds results to Groq (Llama 3) to generate dynamic follow-up queries.
    4. Returns everything to fuel the frontend graph view.
    """
    # 1. Retrieve Vectors
    query_vector = get_embedding(q, is_query=True)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    formatted_results = []
    context_strings = []
    
    # 2. Format Results & Build Context string for the LLM
    for match in results.matches:
        meta = match.metadata
        formatted_results.append(
            ResourceResponse(
                id=match.id,
                title=meta["title"],
                url=meta["url"],
                description=meta.get("description", ""),
                tags=meta.get("tags", [])
            )
        )
        context_strings.append(f"Title: {meta['title']} - Tags: {', '.join(meta.get('tags', []))}")
        
    context_block = "\n".join(context_strings)
    
    # 3. Generate Smart Queries based on the actual database hits
    smart_queries = generate_smart_queries(q, context_block)
    
    return SearchResponse(
        results=formatted_results,
        ai_suggested_queries=smart_queries
    )

@app.delete("/resources/{resource_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_resource(resource_id: str):
    """Deletes a resource from the vector index by its ID."""
    try:
        # Pinecone accepts a list of IDs to delete
        index.delete(ids=[resource_id])
        return 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete from Pinecone: {str(e)}")