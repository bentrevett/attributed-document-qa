import datasets
import sentence_transformers

def get_embeddings(batch, model):
    embeddings = model.encode(batch["text"])
    return {"embeddings": embeddings}

model = sentence_transformers.SentenceTransformer(
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
)

dataset = datasets.load_dataset("json", data_files="data.jsonl", split="train")
dataset = dataset.map(get_embeddings, batched=True, batch_size=32, fn_kwargs={"model": model})
dataset = dataset.with_format(
    type="numpy", columns=["embeddings"], output_all_columns=True,
)
dataset.add_faiss_index("embeddings")
dataset.save_faiss_index("embeddings", "index.faiss")