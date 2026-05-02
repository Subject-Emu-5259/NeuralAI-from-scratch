#!/usr/bin/env python3
"""
NeuralAI DPO Expansion Script
Generates a large preference dataset for Phase 3 Alignment
"""

import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add NeuralAI root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_dpo import DPODatasetBuilder

def expand_dpo():
    builder = DPODatasetBuilder(output_path="data/train_dpo_expanded.jsonl")
    
    # 1. Base pairs from existing builder
    builder.generate_code_pairs()
    builder.generate_response_pairs()
    builder.generate_safety_pairs()
    builder.generate_tool_pairs()
    
    # 2. Add New: Advanced Coding Pairs
    builder.add_pair(
        prompt="Write a Python function to fetch data from an API with retries",
        chosen="""import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def fetch_with_retries(url):
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session.get(url).json()""",
        rejected="""import requests
import time

def fetch(url):
    for i in range(3):
        try:
            return requests.get(url).json()
        except:
            time.sleep(1)
    return None""",
        category="robustness"
    )
    
    builder.add_pair(
        prompt="Explain the difference between a list and a tuple in Python",
        chosen="Lists are mutable (can be changed) and use more memory. Tuples are immutable (fixed) and more memory-efficient. Use lists for dynamic data and tuples for fixed records.",
        rejected="Lists have brackets [] and tuples have parentheses (). You can change lists but not tuples.",
        category="depth"
    )
    
    # 3. Add New: Tool Usage Precision
    builder.add_pair(
        prompt="List all files in the current directory including hidden ones",
        chosen="I'll list all files for you:\n\n```bash\n$ ls -la\n```",
        rejected="You can use the `ls` command to see your files.",
        category="tool_precision"
    )
    
    # 4. Add New: RAG/Context Alignment
    builder.add_pair(
        prompt="Based on the uploaded document, what is the company's revenue?",
        chosen="According to the document 'Annual_Report.pdf', the company's revenue for 2025 was $4.2 billion, a 12% increase from the previous year.",
        rejected="The company is doing well and made billions of dollars last year.",
        category="grounding"
    )

    # 5. Add New: Refactoring Alignment
    builder.add_pair(
        prompt="Refactor this to use list comprehension: result = []\nfor x in range(10):\n    if x % 2 == 0:\n        result.append(x * 2)",
        chosen="```python\nresult = [x * 2 for x in range(10) if x % 2 == 0]\n```",
        rejected="```python\n# You can do it like this\nresult = list(map(lambda x: x * 2, filter(lambda x: x % 2 == 0, range(10))))\n```",
        category="idiomatic_code"
    )

    # Save expanded dataset
    builder.build_all()
    print(f"Total pairs generated: {len(builder.pairs)}")

if __name__ == "__main__":
    expand_dpo()
