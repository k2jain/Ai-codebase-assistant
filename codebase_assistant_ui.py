import os
import ast
import shutil
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
import gradio as gr
from git import Repo
from sentence_transformers import SentenceTransformer

ALLOWED_EXTENSIONS = {".py"}


@dataclass
class CodeNode:
    file_path: str
    node_type: str
    name: str
    start_line: int
    end_line: int
    content: str
    summary_text: str


def clone_repo(repo_url: str, target_dir: str = "cloned_repo") -> str:
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    Repo.clone_from(repo_url, target_dir)
    return target_dir


def read_python_files(root_dir: str) -> List[str]:
    file_paths = []
    for root, _, files in os.walk(root_dir):
        if ".git" in root:
            continue
        for file in files:
            if os.path.splitext(file)[1] in ALLOWED_EXTENSIONS:
                file_paths.append(os.path.join(root, file))
    return file_paths


def get_source_segment(lines: List[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1:end])


def parse_python_file(file_path: str) -> List[CodeNode]:
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    nodes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is None:
                continue

            content = get_source_segment(lines, start, end)
            summary_text = (
                f"Function {node.name} in {file_path}. "
                f"Python function definition with code:\n{content}"
            )

            nodes.append(CodeNode(
                file_path=file_path,
                node_type="function",
                name=node.name,
                start_line=start,
                end_line=end,
                content=content,
                summary_text=summary_text,
            ))

        elif isinstance(node, ast.ClassDef):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is None:
                continue

            content = get_source_segment(lines, start, end)
            summary_text = (
                f"Class {node.name} in {file_path}. "
                f"Python class definition with code:\n{content}"
            )

            nodes.append(CodeNode(
                file_path=file_path,
                node_type="class",
                name=node.name,
                start_line=start,
                end_line=end,
                content=content,
                summary_text=summary_text,
            ))

    if not nodes:
        nodes.append(CodeNode(
            file_path=file_path,
            node_type="file",
            name=os.path.basename(file_path),
            start_line=1,
            end_line=len(lines),
            content=source,
            summary_text=f"Full file {file_path}. Code:\n{source}",
        ))

    return nodes


def explain_match(query: str, node: CodeNode) -> str:
    q = query.lower()
    text = node.content.lower()
    reasons = []

    if "login" in q or "auth" in q or "authentication" in q:
        if "login" in node.name.lower() or "auth" in node.name.lower():
            reasons.append("symbol name matches the authentication intent")
        if "password" in text or "token" in text:
            reasons.append("code contains credential or token handling")

    if "error" in q or "keyerror" in q or "bug" in q:
        if "[" in node.content and "]" in node.content:
            reasons.append("code accesses dictionary-style fields that may trigger missing-key bugs")
        if "raise" in text:
            reasons.append("code contains explicit error raising logic")

    if "user" in q and "user" in text:
        reasons.append("code directly references user-related fields or objects")

    if not reasons:
        reasons.append("semantic embedding similarity matched this code to the query")

    return "; ".join(reasons)


class CodebaseAssistant:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.nodes: List[CodeNode] = []
        self.index = None
        self.repo_root = None

    def build_index(self, root_dir: str):
        self.repo_root = root_dir
        file_paths = read_python_files(root_dir)

        all_nodes = []
        for path in file_paths:
            all_nodes.extend(parse_python_file(path))

        if not all_nodes:
            raise ValueError("No valid Python code found.")

        self.nodes = all_nodes
        texts = [n.summary_text for n in self.nodes]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        return f"Indexed {len(self.nodes)} code nodes from {len(file_paths)} files."

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.index is None:
            raise ValueError("Index not built yet.")

        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_embedding, min(top_k * 4, len(self.nodes)))

        results = []
        seen = set()

        for rank_idx, idx in enumerate(indices[0]):
            node = self.nodes[idx]
            unique_key = (node.file_path, node.name, node.start_line, node.end_line)
            if unique_key in seen:
                continue
            seen.add(unique_key)

            results.append({
                "rank": len(results) + 1,
                "file_path": node.file_path,
                "node_type": node.node_type,
                "name": node.name,
                "start_line": node.start_line,
                "end_line": node.end_line,
                "content": node.content,
                "distance": float(distances[0][rank_idx]),
                "explanation": explain_match(query, node),
            })

            if len(results) >= top_k:
                break

        return pd.DataFrame(results)

    def format_answer(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k=top_k)

        lines = [f"Question: {query}", "", "Top relevant code nodes:"]
        for _, row in results.iterrows():
            lines.append(
                f"\n[{row['rank']}] {row['node_type'].upper()} `{row['name']}` "
                f"in {row['file_path']} (lines {row['start_line']}-{row['end_line']})"
            )
            lines.append(f"Why it matched: {row['explanation']}")
            lines.append(row["content"])
            lines.append("-" * 90)

        return "\n".join(lines)


assistant = CodebaseAssistant()


def index_local_repo(local_path: str):
    try:
        message = assistant.build_index(local_path)
        return message
    except Exception as e:
        return f"Error: {str(e)}"


def index_github_repo(repo_url: str):
    try:
        repo_path = clone_repo(repo_url, "cloned_repo")
        message = assistant.build_index(repo_path)
        return f"Cloned {repo_url}\n{message}"
    except Exception as e:
        return f"Error: {str(e)}"


def ask_codebase(query: str, top_k: int):
    try:
        answer = assistant.format_answer(query, top_k=top_k)
        table = assistant.search(query, top_k=top_k)[
            ["rank", "file_path", "node_type", "name", "start_line", "end_line", "explanation", "distance"]
        ]
        return answer, table
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()


with gr.Blocks(title="AI Codebase Search & Debug Assistant") as demo:
    gr.Markdown("# AI Codebase Search & Debug Assistant")
    gr.Markdown("Index a local Python repo or clone a public GitHub repo, then ask natural-language questions about the codebase.")

    with gr.Tab("Index Local Repo"):
        local_repo_path = gr.Textbox(label="Local Folder Path", value="sample_repo")
        local_index_btn = gr.Button("Index Local Repo")
        local_status = gr.Textbox(label="Status", lines=3)
        local_index_btn.click(fn=index_local_repo, inputs=local_repo_path, outputs=local_status)

    with gr.Tab("Clone GitHub Repo"):
        github_repo_url = gr.Textbox(
            label="Public GitHub Repo URL",
            placeholder="https://github.com/username/repo"
        )
        github_index_btn = gr.Button("Clone and Index Repo")
        github_status = gr.Textbox(label="Status", lines=3)
        github_index_btn.click(fn=index_github_repo, inputs=github_repo_url, outputs=github_status)

    with gr.Tab("Ask Questions"):
        query_box = gr.Textbox(
            label="Ask about the codebase",
            placeholder="Where is authentication handled? / Why might username throw a KeyError?"
        )
        top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top K Results")
        ask_btn = gr.Button("Search Codebase")
        answer_box = gr.Textbox(label="Assistant Output", lines=28)
        results_table = gr.Dataframe(label="Retrieved Results")
        ask_btn.click(fn=ask_codebase, inputs=[query_box, top_k_slider], outputs=[answer_box, results_table])

demo.launch(share=True)
