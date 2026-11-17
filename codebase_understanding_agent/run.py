#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace

from app.config import AgentConfig
from app.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地只读代码库理解 Agent。")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--port", type=int, help="覆盖本地监听端口（默认 8765）")
    args = parser.parse_args()

    config = AgentConfig.load()
    if args.port is not None:
        if not 1024 <= args.port <= 65535:
            parser.error("--port 必须在 1024–65535 之间")
        config = replace(config, port=args.port)
    serve(config, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
