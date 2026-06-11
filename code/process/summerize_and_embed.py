import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_summarization_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "facebook/bart-large-cnn"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    def summarize(text):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    return summarize

def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    return model

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '../../data/raw/test_idea.csv')
    output_path = os.path.join(script_dir, '../../data/intermediate/test_ideas_summarized.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)
    texts = df['idea_text'].fillna('').astype(str).tolist()

    print(f"Loading models...")
    summarize = load_summarization_model()
    embedding_model = load_embedding_model()

    print(f"Summarizing {len(texts)} ideas...")
    summaries = []
    for text in tqdm(texts):
        # Truncate to 2000 chars to fit the model's context window
        short_text = text[:2000]
        summary = summarize(short_text)
        summaries.append(summary)

    df['summary'] = summaries
    df.to_csv(output_path, index=False)
    print(f"Summarization complete. Output saved to {output_path}")

    print("Generating embeddings for summaries...")
    embeddings = embedding_model.encode(summaries, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Add the first few embedding values as columns for inspection
    emb_df = pd.DataFrame(embeddings[:, :5], columns=['emb_0', 'emb_1', 'emb_2', 'emb_3', 'emb_4'])
    result_df = pd.concat([df[['id', 'idea_text', 'summary']], emb_df], axis=1)
    result_path = os.path.join(os.path.dirname(output_path), 'test_ideas_with_summaries_and_embeddings.csv')
    result_df.to_csv(result_path, index=False)
    print(f"Data with summaries and first 5 embedding dimensions saved to {result_path}")

if __name__ == '__main__':
    main()
