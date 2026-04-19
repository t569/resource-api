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

# Load environment variables for local development
load_dotenv()

app = FastAPI(
    title="Digital Garden RAG API", 
    description="Vector Search Pipeline with LLM-Augmented Follow-up Queries"
)

# --- Middleware & Security ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "https://t569.github.io"
        "https://resource-api-eight.vercel.app" # Your actual frontend domains go here
    ],
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
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "resource-api-garden")) 
EMBEDDING_MODEL = "llama-text-embed-v2"

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
    input_type = "query" if is_query else "passage"
    response = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[text],
        parameters={"input_type": input_type, "truncate": "END"}
    )
    return response[0].values

def generate_smart_queries(search_term: str, retrieved_context: str) -> List[str]:
    prompt = f"""
    The user searched for: "{search_term}".
    Based on our database, we found these relevant resources:
    {retrieved_context}
    
    Based on what they searched and what we actually have in our database, generate exactly 3 highly specific, 
    advanced search queries they could use next to explore deeper into these specific niches. 
    
    Return a JSON object with a single key "queries" that contains the array of 3 strings. No formatting.
    """
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", 
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        response_text = chat_completion.choices[0].message.content
        parsed = json.loads(response_text)
        
        if isinstance(parsed, dict) and "queries" in parsed:
            return parsed["queries"]
        elif isinstance(parsed, dict):
            return next(iter(parsed.values()))
            
        return parsed
    except Exception as e:
        print(f"LLM Generation failed: {e}")
        return [f"Advanced {search_term} architecture", f"{search_term} edge cases"]


def standardize_ontology(raw_tags: List[str], description: str) -> List[str]:
    """Passes raw user tags to the LLM to map them to the unified digital garden ontology."""
    
    prompt = f"""
    You are an ontology middleware for a systems engineering knowledge graph.
    The user is injecting a resource with the description: "{description}"
    The user provided these raw tags: {raw_tags}
    
    1. Consolidate aliases (e.g., 'k8s' -> 'kubernetes', 'operating-systems' -> 'os').
    2. Format tags hierarchically using a forward slash where appropriate (e.g., 'infrastructure/docker', 'cryptography/zkp').
    3. Keep it strictly technical and concise. Limit to 5 tags maximum.
    
    Return ONLY a raw JSON object with a single key "clean_tags" containing an array of strings.
    """
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", 
            temperature=0.1, # Low temperature for highly deterministic categorization
            response_format={"type": "json_object"}
        )
        parsed = json.loads(chat_completion.choices[0].message.content)
        return parsed.get("clean_tags", raw_tags) # Fallback to raw tags if missing
    except Exception as e:
        print(f"Ontology mapping failed: {e}")
        return raw_tags # Failsafe: if the LLM crashes, just use the user's original inputs
    

# --- Endpoints ---

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Digital Garden RAG API",
        "status": "Online",
        "docs": "/docs"
    }

# TODO: we might need to find a way to prevent duplicate resource injections (e.g., same URL) - maybe a quick metadata-only search before allowing an injection to proceed? For now, we just let it be and rely on the user not to be spammy.
# in essence for use to save state and not burn through our tokens
@app.post("/resources/inject", status_code=201, dependencies=[Depends(verify_api_key)])
def inject_resource(resource: ResourceSubmission):
    resource_id = str(uuid.uuid4())
    
    # --- ONTOLOGY MIDDLEWARE INTERCEPTS HERE ---
    clean_tags = standardize_ontology(resource.tags, resource.description)
    
    text_to_embed = f"Title: {resource.title}. Description: {resource.description}. Tags: {', '.join(clean_tags)}."
    vector = get_embedding(text_to_embed, is_query=False)
    
    metadata = {
        "title": resource.title,
        "url": str(resource.url),
        "description": resource.description or "",
        "category": resource.category,
        "tags": clean_tags  # Save the LLM's highly structured tags instead
    }
    index.upsert(vectors=[(resource_id, vector, metadata)])
    
    return {"message": "Resource successfully injected.", "id": resource_id}

@app.get("/search", response_model=SearchResponse)
def search_pipeline(q: str = Query(..., description="User search query"), top_k: int = Query(5)):
    query_vector = get_embedding(q, is_query=True)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    formatted_results = []
    context_strings = []
    
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
    smart_queries = generate_smart_queries(q, context_block)
    
    return SearchResponse(
        results=formatted_results,
        ai_suggested_queries=smart_queries
    )

@app.get("/graph/cluster")
def get_graph_cluster():
    """Generates a structured node/link knowledge graph from the vector DB."""
    
    # We use a generic vector to pull a broad cluster of the latest/top 50 resources.
    # (Llama-text-embed-v2 expects text, so we embed a broad tech string).
    generic_vector = get_embedding("software systems architecture embedded engineering data", is_query=True)
    
    # Fetch top 50 resources to form the graph
    results = index.query(vector=generic_vector, top_k=50, include_metadata=True)
    
    nodes = []
    links = []
    tags_map = set()

    for match in results.matches:
        meta = match.metadata
        resource_id = match.id
        
        # 1. Add the Resource Node
        nodes.append({
            "id": resource_id,
            "name": meta["title"],
            "url": meta["url"],
            "group": "resource"
        })

        # 2. Add Tag Nodes and create Links
        for tag in meta.get("tags", []):
            tag_id = f"tag-{tag.lower()}"
            
            # Ensure we only add each tag node once
            if tag_id not in tags_map:
                tags_map.add(tag_id)
                nodes.append({
                    "id": tag_id,
                    "name": f"#{tag}",
                    "group": "tag"
                })
            
            # Map the connection (Edge)
            links.append({
                "source": resource_id,
                "target": tag_id
            })

    return {"nodes": nodes, "links": links}


@app.delete("/resources/{resource_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_resource(resource_id: str):
    try:
        index.delete(ids=[resource_id])
        return 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete from Pinecone: {str(e)}")