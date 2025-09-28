import faiss
import torch
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 1. Define model path and configuration
#    Replace with the actual path to your model
model_path = "/path/to/your/Youtu-Embedding-V1_sentence_transformers"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Instantiate the HuggingFaceEmbedding model
#    This model supports custom instructions.
#    - For asymmetric tasks (e.g., Information Retrieval), provide different instructions
#      for queries and documents.
#    - For symmetric tasks (e.g., Semantic Textual Similarity), provide the same
#      instruction for both `query_instruction` and `text_instruction`.
embeddings = HuggingFaceEmbedding(
    model_name=model_path,
    trust_remote_code=True,
    device=device,
    # ❗️❗️❗️ Replace these placeholders with your model's actual instructions ❗️❗️❗️
    query_instruction="[Your query instruction here]",  
    text_instruction="[Your document instruction here]"
)

# 3. Prepare sample data
data = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# Create TextNode objects for LlamaIndex
nodes = [TextNode(id_=str(i), text=text) for i, text in enumerate(data)]

# Generate embeddings for each node
# The `get_text_embedding` method will automatically apply the `text_instruction`
for node in nodes:
    node.embedding = embeddings.get_text_embedding(node.get_content())

# 4. Create and build the FAISS vector store
embed_dim = len(nodes[0].embedding)
store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(embed_dim))
store.add(nodes)

# 5. Perform a similarity search
query = "Which planet is known as the Red Planet?"
# The `get_query_embedding` method will automatically apply the `query_instruction`
query_embedding = embeddings.get_query_embedding(query)

results = store.query(
    VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3)
)

# 6. Print results
print(f"Query: {query}\n")
print("Results:")
for idx, score in zip(results.ids, results.similarities):
    print(f"- Text: {data[int(idx)]} (Score: {score:.4f})")
