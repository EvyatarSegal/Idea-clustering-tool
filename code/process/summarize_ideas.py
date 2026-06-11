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

def summarize(text, model_name, max_retries=1):
    # Clean and truncate
    text = text.strip().replace('\n', ' ').replace('\r', '')
    if len(text) > 1500:
        text = text[:1500]  # truncate to avoid context overflow
    
    prompt = f"Summarize in one short sentence (max 20 words): {text}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 60
    }
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content:
                    return content
                else:
                    print(f"Attempt {attempt+1}: empty content for text starting: {text[:50]}...")
            else:
                print(f"HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
        time.sleep(1)
    
    # Fallback: return a truncated version of the original
    fallback = text[:80] + "..." if len(text) > 80 else text
    print(f"Using fallback summary: {fallback}")
    return fallback
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
