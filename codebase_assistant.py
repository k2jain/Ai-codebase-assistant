import os
import ast
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
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


def read_python_files(root_dir: str) -> List[str]:
    file_paths = []
    for root, _, files in os.walk(root_dir):
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
            summary_text = f"Function {node.name} in {file_path}. Python function definition. Code:\n{content}"

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
            summary_text = f"Class {node.name} in {file_path}. Python class definition. Code:\n{content}"

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


class CodebaseAssistant:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.nodes: List[CodeNode] = []
        self.index = None

    def build_index(self, root_dir: str):
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

        print(f"Indexed {len(self.nodes)} code nodes from {len(file_paths)} files.")

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.index is None:
            raise ValueError("Index not built yet.")

        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_embedding, min(top_k * 3, len(self.nodes)))

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
            })

            if len(results) >= top_k:
                break

        return pd.DataFrame(results)

    def answer_question(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k=top_k)

        lines = [f"Question: {query}", "", "Top relevant code nodes:"]
        for _, row in results.iterrows():
            lines.append(
                f"\n[{row['rank']}] {row['node_type'].upper()} `{row['name']}` "
                f"in {row['file_path']} (lines {row['start_line']}-{row['end_line']})\n"
                f"{row['content']}\n" + "-" * 80
            )
        return "\n".join(lines)

    def debug_error(self, error_message: str, top_k: int = 5) -> str:
        query = f"Find code likely related to this error: {error_message}"
        return self.answer_question(query, top_k=top_k)


if __name__ == "__main__":
    assistant = CodebaseAssistant()
    repo_path = "sample_repo"
    assistant.build_index(repo_path)

    print("\n" + "=" * 100)
    print(assistant.answer_question("Where is authentication handled?"))

    print("\n" + "=" * 100)
    print(assistant.answer_question("Where is user login implemented?"))

    print("\n" + "=" * 100)
    print(assistant.debug_error("KeyError: username in login request"))
