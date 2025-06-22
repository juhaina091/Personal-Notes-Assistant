import os
import faiss
import numpy as np
from utils import load_documents, split_chunks, embed_documents
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =======================
# Load Models
# =======================
# Sentence embedding model (lightweight)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Local language model for generation (FLAN-T5)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# =======================
# Step 1: Load and prepare documents
# =======================
print("Loading documents...")
documents = load_documents("data")
chunks = []
for doc in documents:
    chunks.extend(split_chunks(doc))

print(f"Total chunks: {len(chunks)}")

# =======================
# Step 2: Generate embeddings
# =======================
print("Embedding...")
embeddings = embed_model.encode(chunks, show_progress_bar=True)

# =======================
# Step 3: Index with FAISS
# =======================
print("Indexing...")
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and chunks for future use
os.makedirs("index", exist_ok=True)
faiss.write_index(index, "index/notes.index")
with open("index/chunks.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

print("Setup complete. You can now ask questions!")

# =======================
# Step 4: Query function
# =======================
def query_notes(question, top_k=3):
    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed), top_k)

    # Load chunks again (if you separate build/query phases, keep this step)
    with open("index/chunks.txt", "r", encoding="utf-8") as f:
        all_chunks = f.read().splitlines()
    
    context = "\n\n".join([all_chunks[i] for i in I[0]])
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

# =======================
# Step 5: Simple CLI
# =======================
while True:
    q = input("\n📝 Ask a question (or type 'exit'): ")
    if q.lower() == "exit":
        break
    answer = query_notes(q)
    print(f"\n🔍 Answer:\n{answer}")
