import json
from pathlib import Path
from datetime import datetime

class DPODatasetBuilder:
    def __init__(self, output_path="data/train_dpo_expanded.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pairs = []
    
    def add_pair(self, prompt, chosen, rejected, category="general"):
        self.pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "category": category,
            "created": datetime.now().isoformat()
        })
    
    def build_all(self):
        with open(self.output_path, "w") as f:
            for pair in self.pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"Saved {len(self.pairs)} pairs to {self.output_path}")

def generate():
    builder = DPODatasetBuilder(output_path="/home/workspace/Projects/NeuralAI/data/train_dpo_expanded.jsonl")
    
    # Coding
    builder.add_pair(
        "Write a function to reverse a string",
        "def reverse_string(s: str) -> str:\n    return s[::-1]",
        "def reverse_string(s):\n    # TODO: implement\n    pass",
        "code_correctness"
    )
    
    # API with retries
    builder.add_pair(
        "Write a Python function to fetch data from an API with retries",
        "import requests\nfrom requests.adapters import HTTPAdapter\nfrom urllib3.util.retry import Retry\n\ndef fetch_with_retries(url):\n    session = requests.Session()\n    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])\n    adapter = HTTPAdapter(max_retries=retry)\n    session.mount('http://', adapter)\n    session.mount('https://', adapter)\n    return session.get(url).json()",
        "import requests\nimport time\n\ndef fetch(url):\n    for i in range(3):\n        try:\n            return requests.get(url).json()\n        except:\n            time.sleep(1)\n    return None",
        "robustness"
    )
    
    # List vs Tuple
    builder.add_pair(
        "Explain the difference between a list and a tuple in Python",
        "Lists are mutable (can be changed) and use more memory. Tuples are immutable (fixed) and more memory-efficient. Use lists for dynamic data and tuples for fixed records.",
        "Lists have brackets [] and tuples have parentheses (). You can change lists but not tuples.",
        "depth"
    )

    # Tool precision
    builder.add_pair(
        "List all files in the current directory including hidden ones",
        "I'll list all files for you:\n\n```bash\n$ ls -la\n```",
        "You can use the `ls` command to see your files.",
        "tool_precision"
    )

    # RAG Grounding
    builder.add_pair(
        "Based on the uploaded document, what is the company's revenue?",
        "According to the document 'Annual_Report.pdf', the company's revenue for 2025 was $4.2 billion, a 12% increase from the previous year.",
        "The company is doing well and made billions of dollars last year.",
        "grounding"
    )

    builder.build_all()

if __name__ == "__main__":
    generate()
