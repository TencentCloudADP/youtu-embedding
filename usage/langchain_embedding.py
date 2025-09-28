import torch
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# 1. Define model path and device
#    Replace with the actual path to your model
model_path = "/path/to/your/Youtu-Embedding-V1_sentence_transformers"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Define arguments for the underlying SentenceTransformer model
model_kwargs = {
    'trust_remote_code': True,
    'device': device
}

# 3. Instantiate HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
)

# 4. Define your instructions separately
#    ❗️❗️❗️ Replace these placeholders with your model's actual instructions ❗️❗️❗️
query_instruction = "[Your query instruction here]"
embed_instruction = "[Your document instruction here]"

# 5. Prepare data and manually prepend instructions
data = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# Create Document objects
documents = [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(data)]

# ❗️ Manually prepend the document instruction to the content of each Document
for doc in documents:
    doc.page_content = embed_instruction + doc.page_content

# 6. Create the vector store
#    The documents passed to this method now contain the prepended instructions.
vector_store = FAISS.from_documents(documents, embedder, distance_strategy="MAX_INNER_PRODUCT")

# 7. Perform a similarity search
query = "Which planet is known as the Red Planet?"
# ❗️ Manually prepend the query instruction to the query string
instructed_query = query_instruction + query
results = vector_store.similarity_search_with_score(instructed_query, k=3)

# 8. Print the results
print(f"Original Query: {query}\n")
print("Results:")
for doc, score in results:
    # Remove the instruction prefix from the document content for a cleaner output
    original_content = doc.page_content.replace(embed_instruction, "", 1)
    print(f"- Text: {original_content} (Score: {score:.4f})")
