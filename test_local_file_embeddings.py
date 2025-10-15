"""
Local Markdown embedding + interactive search with file selection:

- 仅检索指定目录（默认 ./test-docs）
- 启动时弹出文件选择菜单（方向键上下，回车确认），可选 All files 或单个 md 文件
- 构建段向量后进入交互检索：输入 query 返回 Top-K 匹配
"""

import os
import re
import json
import argparse
import sys
from typing import List, Dict, Tuple, Iterable
from pathlib import Path

import numpy as np

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Reuse the local embedding model
from test_transformers_local import LLMEmbeddingModel

# --- Markdown segmentation utilities ---
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
YAML_FRONTMATTER_PATTERN = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
CODEBLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；.!?;])\s+")
MD_EXTENSIONS = {".md", ".mdx", ".markdown"}


def list_markdown_files(input_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in MD_EXTENSIONS:
                files.append(os.path.join(root, fn))
    return sorted(files)


def strip_frontmatter(text: str) -> str:
    m = YAML_FRONTMATTER_PATTERN.match(text)
    if m:
        return text[m.end():]
    return text


def extract_sections(md_text: str) -> List[Dict]:
    sections: List[Dict] = []
    matches = list(HEADING_PATTERN.finditer(md_text))
    if not matches:
        sections.append({"title": "", "level": 0, "content": md_text})
        return sections

    first_start = matches[0].start()
    if first_start > 0:
        leading = md_text[:first_start].strip()
        if leading:
            sections.append({"title": "", "level": 0, "content": leading})

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end].strip()
        sections.append({"title": title, "level": level, "content": content})

    return sections


def split_sentences(text: str) -> List[str]:
    parts: List[str] = []
    last_end = 0
    for m in CODEBLOCK_PATTERN.finditer(text):
        before = text[last_end:m.start()]
        if before.strip():
            parts.extend([s for s in SENTENCE_SPLIT_PATTERN.split(before.strip()) if s.strip()])
        code = text[m.start():m.end()]
        parts.append(code)
        last_end = m.end()
    after = text[last_end:]
    if after.strip():
        parts.extend([s for s in SENTENCE_SPLIT_PATTERN.split(after.strip()) if s.strip()])
    return parts


def group_by_tokens(sentences: List[str], tokenizer, max_tokens: int, reserve_tokens: int = 16) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0
    budget = max(8, max_tokens - reserve_tokens)

    def count_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=True, truncation=True, max_length=max_tokens))

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if sent_tokens >= budget:
            if buf:
                chunks.append("\n".join(buf).strip())
                buf, buf_tokens = [], 0
            chunks.append(sent.strip())
            continue

        if buf_tokens + sent_tokens > budget:
            if buf:
                chunks.append("\n".join(buf).strip())
            buf, buf_tokens = [sent], sent_tokens
        else:
            buf.append(sent)
            buf_tokens += sent_tokens

    if buf:
        chunks.append("\n".join(buf).strip())

    return [c for c in chunks if c.strip()]


def segment_markdown(md_text: str, tokenizer, max_length: int) -> List[Dict]:
    segments: List[Dict] = []
    sections = extract_sections(md_text)
    idx = 0
    for sec in sections:
        sentences = split_sentences(sec["content"])
        grouped = group_by_tokens(sentences, tokenizer, max_tokens=max_length)
        for g in grouped:
            segments.append({
                "title": sec["title"],
                "level": sec["level"],
                "text": g,
                "segment_index": idx
            })
            idx += 1
    return segments


def encode_segments(model: LLMEmbeddingModel, segments: List[Dict], batch_size: int) -> List[List[float]]:
    texts = [seg["text"] for seg in segments]
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        reps = model.encode_passages(batch)
        embs = reps.detach().cpu().numpy().tolist()
        embeddings.extend(embs)
    return embeddings


# --- Interactive menu (arrow keys) ---
def select_files_menu(files: List[str]) -> List[str]:
    """
    Show a curses-based menu to select either 'All files' or a single file.
    Up/Down to move, Enter to confirm. Returns list of files to process.
    Fallback to numbered input if curses fails or is not in a TTY.
    """
    # Normalize to relative paths for display
    display_items = ["All files"] + [os.path.relpath(f) for f in files]

    def curses_menu(stdscr):
        import curses
        curses.curs_set(0)
        stdscr.keypad(True)
        h, w = stdscr.getmaxyx()
        idx = 0

        title = "请选择要处理的文档（方向键上下，回车确认）："
        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, title)
            for i, item in enumerate(display_items):
                prefix = "➤ " if i == idx else "  "
                line = f"{prefix}{item}"
                if i == idx:
                    stdscr.attron(curses.A_REVERSE)
                    stdscr.addstr(2 + i, 2, line[:w-4])
                    stdscr.attroff(curses.A_REVERSE)
                else:
                    stdscr.addstr(2 + i, 2, line[:w-4])
            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord('k')):
                idx = (idx - 1) % len(display_items)
            elif key in (curses.KEY_DOWN, ord('j')):
                idx = (idx + 1) % len(display_items)
            elif key in (curses.KEY_ENTER, 10, 13):
                return idx

    use_curses = sys.stdin.isatty() and sys.stdout.isatty()
    if use_curses:
        try:
            import curses  # noqa
            selected_index = __import__("curses").wrapper(curses_menu)
        except Exception:
            selected_index = None
    else:
        selected_index = None

    if selected_index is None:
        # Fallback numbered selection
        print("请选择要处理的文档：")
        for i, item in enumerate(display_items):
            print(f"  [{i}] {item}")
        while True:
            choice = input("输入数字选择（回车默认 0=All files）：").strip()
            if choice == "":
                selected_index = 0
                break
            if choice.isdigit():
                ci = int(choice)
                if 0 <= ci < len(display_items):
                    selected_index = ci
                    break
            print("无效选择，请重试。")

    if selected_index == 0:
        return files
    else:
        return [files[selected_index - 1]]


# --- Similarity and search UI ---
def interactive_search(model: LLMEmbeddingModel, emb_np: np.ndarray, meta: List[Dict], top_k: int):
    print("\n[SEARCH] 请输入查询（直接回车退出）:")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            break
        if not query:
            print("[EXIT] 已退出。")
            break

        q_reps = model.encode_queries(query)
        q_vec = q_reps.detach().cpu().numpy().squeeze(0)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)

        scores = emb_np @ q_norm  # cosine for normalized embeddings
        k = min(max(top_k, 1), len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        print("\n" + "=" * 80)
        print(f"🔍 Query: {query}")
        print("=" * 80)
        for rank, idx in enumerate(top_idx, start=1):
            s = float(scores[idx])
            m = meta[idx]
            title = m.get("title", "")
            level = m.get("level", 0)
            level_mark = "#" * level if level > 0 else ""
            file_path = m.get("file_path", "")
            text = m.get("text", "")

            bar_length = max(0, min(30, int(s * 30)))
            score_bar = "█" * bar_length + "░" * (30 - bar_length)
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))

            print(f"\n{medal} Score: {s:.4f} | [{score_bar}]")
            if title:
                print(f"   Section: {level_mark} {title}")
            print(f"   File: {file_path}")
            print(f"   Snippet:\n   \"{text[:500]}\"")
        print("\n" + "=" * 80)
        print("[Tip] 输入新的查询或回车退出。")


def process_file(path: str, model: LLMEmbeddingModel, strip_yaml: bool) -> Tuple[List[Dict], List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if strip_yaml:
        content = strip_frontmatter(content)
    segments = segment_markdown(content, model.tokenizer, model.max_length)
    embeddings = encode_segments(model, segments, model.batch_size)
    # attach file_path in segment meta
    for seg in segments:
        seg["file_path"] = path
    return segments, embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local file Markdown embedding + interactive search with file selection")
    parser.add_argument("--input_dir", type=str, default="./test-docs", help="Root directory to scan for markdown files")
    parser.add_argument("--model_path", type=str, default="./Youtu-Embedding", help="Local embedding model path/name for transformers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per encode_passages call")
    parser.add_argument("--max_length", type=int, default=1024, help="Tokenizer max_length for the underlying model")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA GPU id if available")
    parser.add_argument("--strip_frontmatter", action="store_true", help="Strip YAML frontmatter at beginning of markdown files")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K results to display in search")
    parser.add_argument("--no_select", action="store_true", help="Skip interactive file selection and process all files")
    return parser.parse_args()


def main():
    args = parse_args()

    files = list_markdown_files(args.input_dir)
    if not files:
        print(f"[ERROR] No markdown files found in: {args.input_dir}")
        sys.exit(1)

    # Interactive selection of files
    if args.no_select:
        selected_files = files
        print(f"[INFO] Skipping selection, processing {len(selected_files)} files.")
    else:
        selected_files = select_files_menu(files)
        if not selected_files:
            print("[ERROR] No file selected.")
            sys.exit(1)
        if len(selected_files) == len(files):
            print(f"[SELECT] All files ({len(files)}).")
        else:
            rel = [os.path.relpath(f) for f in selected_files]
            print(f"[SELECT] Single file: {rel[0]}")

    # Initialize model
    model = LLMEmbeddingModel(
        model_name_or_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gpu_id=args.gpu_id
    )

    # Build in-memory index
    all_segments: List[Dict] = []
    all_embeddings: List[List[float]] = []

    iterator: Iterable[str] = selected_files
    if tqdm is not None:
        iterator = tqdm(selected_files, desc="Indexing markdown files")

    for path in iterator:
        try:
            segments, embeddings = process_file(path, model, strip_yaml=args.strip_frontmatter)
            all_segments.extend(segments)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")

    if not all_segments:
        print("[ERROR] No segments built; aborting.")
        sys.exit(1)

    emb_np = np.array(all_embeddings, dtype=np.float32)

    # Interactive search loop
    interactive_search(model, emb_np, all_segments, top_k=args.top_k)


if __name__ == "__main__":
    main()