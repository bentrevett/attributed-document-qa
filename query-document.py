import openai
import sentence_transformers
import datasets
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, required=True)
parser.add_argument("--openai-api-key", type=str, required=True)
args = parser.parse_args()

os.environ["OPENAI_API_KEY"] = args.openai_api_key
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_chatgpt_api(messages, model_name, temperature):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response

def get_prompt_from_evidences(evidences, query):
    prompt = "Here are some lines of text extracted from a financial document. All information extracted is related to the client.\n"
    for i, text in enumerate(evidences["text"], start=1):
        prompt += f"{i}: {text}\n"
    prompt += "\n"
    prompt += "Answer the following question by using the above extracted text only.\n\n"
    prompt += f"{query}"
    return prompt

def evidence_to_messages(evidences, query):
    prompt = get_prompt_from_evidences(evidences, query)
    messages = [
        {"role": "user", "content": prompt},
    ]
    return messages

def get_evidence(query, model, dataset, k=5):
    query_embedding = model.encode(query)
    _, evidence = dataset.get_nearest_examples(
        "embeddings",
        query_embedding,
        k=k,
    )
    return evidence

def get_answer(query, evidence):
    messages = evidence_to_messages(evidence, query)
    completion = call_chatgpt_api(messages, model_name="gpt-3.5-turbo-0301", temperature=0)
    answer = completion["choices"][0]["message"]["content"].strip()
    return answer

model = sentence_transformers.SentenceTransformer(
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
)
dataset = datasets.load_dataset("json", data_files="data.jsonl", split="train")
dataset.load_faiss_index("embeddings", "index.faiss")

evidence = get_evidence(args.query, model, dataset)
answer = get_answer(args.query, evidence)

print("Evidence:")
print(evidence)
print("Answer:")
print(answer)