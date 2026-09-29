from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Create Hugging Face embedding model
# --------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded successfully!")

text = "What is machine learning?"

vector = embeddings.embed_query(text)

print("\nEmbedding test")
print("-------------------------")
print("Input:", text)
print("Vector type:", type(vector))
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])


# --------------------------------------------------
# 3. Test another sentence
# --------------------------------------------------

text2 = "Machine learning is a type of artificial intelligence."

vector2 = embeddings.embed_query(text2)

print("\nSecond embedding")
print("-------------------------")
print("Input:", text2)
print("Vector dimensions:", len(vector2))
print("First 10 values:", vector2[:10])


# --------------------------------------------------
# 4. Test ChromaDB
# --------------------------------------------------

print("\nTesting ChromaDB...")

vectorstore = Chroma(
    collection_name="embedding_test",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add test documents
documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Python is widely used for data science and machine learning.",
    "The capital of France is Paris.",
]

ids = ["test1", "test2", "test3"]

vectorstore.add_texts(
    texts=documents,
    ids=ids
)

print("Documents successfully added to ChromaDB!")


# --------------------------------------------------
# 5. Test similarity search
# --------------------------------------------------

query = "What is artificial intelligence and machine learning?"

results = vectorstore.similarity_search_with_score(
    query,
    k=3
)

print("\nSimilarity Search")
print("-------------------------")
print("Query:", query)

for doc, score in results:
    print("\nDocument:", doc.page_content)
    print("Score:", score)


print("\n✅ Embedding + ChromaDB test completed successfully!")