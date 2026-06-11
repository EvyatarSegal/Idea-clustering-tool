#!/usr/bin/env python3
"""
Send each business idea to LM Studio's local API and get a short summary.
Input: ../../data/raw/ideas.csv
Output: ../../data/intermediate/ideas_summarized.csv
"""

import os
import requests
import pandas as pd
import argparse
import time
from tqdm import tqdm

LM_STUDIO_URL = "http://localhost:1234/v1"

def summarize(text, model_name):
    prompt = f"""Summarize the following business idea in one short sentence (max 20 words). Keep only the core concept, remove marketing fluff, brand names, and extra details.

Idea: {text}

Summary:"""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 60,
        "stream": False
    }
    try:
        response = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        print(f"Error: {e}")
        return text

def main():
    # Default paths relative to script location (code/process/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, '../../data/raw/test_ideas.csv')
    default_output = os.path.join(script_dir, '../../data/intermediate/test_ideas_summarized.csv')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=default_input, help='Input CSV file')
    parser.add_argument('--output', default=default_output, help='Output CSV file')
    parser.add_argument('--text-col', default='idea_text')
    parser.add_argument('--id-col', default='id')
    parser.add_argument('--model', default='llama-8b', help='Model name in LM Studio')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df = pd.read_csv(args.input)
    texts = df[args.text_col].fillna('').astype(str).tolist()
    
    summaries = []
    for txt in tqdm(texts, desc='Summarizing'):
        summ = summarize(txt, args.model)
        summaries.append(summ)
        time.sleep(0.2)
    
    df['summary'] = summaries
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
