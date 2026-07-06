from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def build_index(
    passages_path: str,
    index_dir: str,
    threads: int = 1,
    java_heap: str = "1536m",
    memory_buffer_mb: int = 256,
) -> None:
    input_path = Path(passages_path)
    output_path = Path(index_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-collection", "JsonCollection",
        "-input", str(input_path.parent),
        "-index", str(output_path),
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", str(threads),
        "-memoryBuffer", str(memory_buffer_mb),
        "-storeRaw",
    ]
    wrapper = f"""
import sys
import jnius_config
jnius_config.add_options("-Xms256m", "-Xmx{java_heap}")
from pyserini.pyclass import autoclass
JIndexCollection = autoclass("io.anserini.index.IndexCollection")
JIndexCollection.main({args!r})
"""
    with tempfile.NamedTemporaryFile("w", suffix="_pyserini_index.py", delete=False) as f:
        f.write(wrapper)
        wrapper_path = f.name
    subprocess.run(["python", wrapper_path], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--java-heap", default="1536m")
    parser.add_argument("--memory-buffer-mb", type=int, default=256)
    args = parser.parse_args()
    build_index(args.passages, args.index_dir, args.threads, args.java_heap, args.memory_buffer_mb)


if __name__ == "__main__":
    main()
