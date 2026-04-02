# AI Codebase Search & Debug Assistant

Developer tool for semantic code search and debugging across Python repositories.

## Features
- Parses Python repos into function- and class-level code nodes using AST
- Uses transformer embeddings + FAISS for semantic retrieval
- Supports natural-language code search and bug-trace debugging
- Includes a Gradio UI for indexing and querying codebases
- Can index a local repo or clone a public GitHub repo

## Example Questions
- Where is authentication handled?
- Where is user login implemented?
- Why might username throw a KeyError?

## Tech Stack
Python, AST, SentenceTransformers, FAISS, Gradio, GitPython

## Run
pip install -q sentence-transformers faiss-cpu gradio gitpython
python codebase_assistant_ui.py
