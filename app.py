from flask import Flask, request, jsonify
from google.genai import Client
import pinecone
from google.genai.errors import ClientError
import time
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import re
import unicodedata
import os
import requests

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Initialize Gemini Client ----------------
client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- Initialize Pinecone ----------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "pdf-chat")
## Grok
API_URL = "https://api.groq.com/openai/v1/chat/completions"  # example Grok endpoint
GROK_API_KEY = os.getenv("GROK_API_KEY")

if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,               # Gemini text-embedding-004 returns 768 dim vectors
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

index = pc.Index(INDEX_NAME)


# ---------------- PDF Text Extraction ----------------
def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text.strip()


# ------------------ Helpers ------------------

def sanitize_text(text: str) -> str:
    """Normalize unicode, replace problematic chars, remove control chars, force UTF-8."""
    if not isinstance(text, str):
        text = str(text)

    # Normalize
    text = unicodedata.normalize("NFKD", text)

    # Replace common fancy punctuation and bullets
    replacements = {
        "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
        "\uf0d8": "-", "\u2022": "-",  # PDF bullets and special chars
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove remaining control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Encode/decode to remove any other non-UTF8 bytes
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    return text.strip()


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    """Split text into overlapping chunks."""
    text = text.replace("\n", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks



# ---------------- Gemini Embedding ----------------
def embed_text(text):
    model = "models/text-embedding-004"
    response = client.models.embed_content(
        model=model,
        contents=text,
    )
    return response.embeddings[0].values

def clean_text(s: str):
    """Remove characters that cannot be encoded in Latin-1."""
    return s.encode("latin-1", "ignore").decode("latin-1")

def embed_text_safely(text: str, retries: int = 3):
    """Retry embedding in case of temporary failures."""
    for attempt in range(retries):
        try:
            return embed_text(text)  # your existing embedding function
        except Exception as e:
            print(f"[Embed Retry {attempt+1}] Failed: {e}")
            time.sleep(1)
    raise RuntimeError("Embedding failed after retries.")

def safe_vector_id(name: str, chunk_idx: int) -> str:
    # keep only alphanumeric, dash, underscore
    clean_name = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
    return f"{clean_name}_chunk_{chunk_idx}"


def clean_for_pinecone(text: str) -> str:
    # Replace common problematic Unicode chars with ASCII
    replacements = {
        "•": "-",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "\u2028": " ",  # line separator
        "\u2029": " ",  # paragraph separator
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Remove any remaining non-ASCII chars
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text.strip()


# Upload PDF Endpoint
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "Upload PDF using field name 'pdf'"}), 400

    pdf_file = request.files["pdf"]
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from PDF
    text = extract_pdf_text(pdf_path)
    if not text:
        return jsonify({"error": "No extractable text found"}), 400

    # Clean the whole text
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = clean_for_pinecone(text)

    # Split into chunks
    chunks = chunk_text(text, max_chars=2000, overlap=200)
    vectors = []

    for i, chunk in enumerate(chunks):
        # Sanitize each chunk
        safe_chunk = chunk.encode("utf-8", errors="ignore").decode("utf-8")
        safe_chunk = clean_for_pinecone(safe_chunk)

        try:
            # Get embedding
            emb = embed_text(safe_chunk)

            # Validate dimension
            if len(emb) != 768:
                print(f"[Skip] Chunk {i} has invalid embedding length: {len(emb)}")
                continue

            # Ensure all floats
            emb = [float(x) for x in emb]

            # Safe vector ID
            vector_id = safe_vector_id(pdf_file.filename, i)

            vectors.append({
                "id": vector_id,
                "values": emb,
                "metadata": {"text": safe_chunk, "source": pdf_file.filename}
            })

        except Exception as e:
            print(f"[Error] Embedding failed for chunk {i}: {e}")
            continue

    if not vectors:
        return jsonify({"error": "No valid vectors to upsert"}), 400

    # Upsert vectors into Pinecone
    try:
        index.upsert(vectors=vectors)
    except Exception as e:
        return jsonify({"error": f"Pinecone upsert failed: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "indexed_chunks": len(vectors),
        "file": pdf_file.filename
    })





# ---------------- Chat with PDF Endpoint ----------------
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Embed question
    q_emb = embed_text(question)  # your existing embedding function

    # Query Pinecone
    result = index.query(
        vector=q_emb,
        top_k=5,
        include_metadata=True
    )

    retrieved_chunks = [match["metadata"]["text"] for match in result["matches"]]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Prepare messages for Groq chat API
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are a helpful PDF assistant. Answer using only the context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        }
    ]

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
        "temperature": 1,
        "max_completion_tokens": 8192,
        "top_p": 1,
        "stream": False,  # set True only if you handle streaming
        "reasoning_effort": "medium"
    }

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        answer_text = data["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"error": f"Groq API request failed: {e}"}), 500

    return jsonify({
        "answer": answer_text,
        "sources": [m["metadata"]["source"] for m in result["matches"]]
    })

@app.route('/')
def home():
    return "Chat with your PDF API (Pinecone version) is running!"


# ---------------- Run Flask ----------------
if __name__ == '__main__':
    app.run(debug=True)
