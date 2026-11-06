"""
One-shot script to install all project deps into the Python 3.12 venv after
torch+cu121 finishes downloading.

Run from project root:
    c:\MDA\multimodal-document-ai\.venv312\Scripts\python.exe scripts/setup_venv312.py
"""
import subprocess, sys

PIP = r"c:\MDA\multimodal-document-ai\.venv312\Scripts\pip.exe"

packages = [
    "datasets>=2.18.0",
    "scikit-learn>=1.4.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "jiwer>=3.0.0",
    "rapidfuzz>=3.0.0",
    "pyyaml>=6.0.0",
    "tqdm>=4.66.0",
    "pytest>=8.0.0",
    "transformers>=4.38.0",
    "seqeval>=1.2.2",
    "accelerate>=0.27.0",
    "sentencepiece>=0.2.0",
    "timm>=0.9.0",
    "safetensors>=0.4.0",
]

print("Installing project dependencies into .venv312...")
result = subprocess.run(
    [PIP, "install"] + packages,
    capture_output=False
)
sys.exit(result.returncode)
