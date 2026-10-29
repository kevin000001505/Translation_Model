"""Command line entry points. The UI runs long jobs through these, as subprocesses."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _log(msg: str) -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="subtune")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="build the list of titles that have subtitles in both languages")
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--refresh", action="store_true", help="rescan the primary language for new subtitles")

    p = sub.add_parser("prepare", help="download (within quota) and align subtitles for features")
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--features-file", required=True, help="JSON list of feature ids, in priority order")

    p = sub.add_parser("train", help="run a training job from a run config JSON")
    p.add_argument("--config", required=True)

    sub.add_parser("quota", help="show the OpenSubtitles download quota")
    sub.add_parser("check-qlora", help="check whether 4-bit QLoRA works on the local GPU")

    p = sub.add_parser("ui", help="launch the Gradio app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.cmd == "ui":
        from .ui import launch

        launch(args.host, args.port, args.share)
        return 0
    if args.cmd == "check-qlora":
        from .train import qlora_supported

        ok, msg = qlora_supported()
        print(json.dumps({"ok": ok, "message": msg}))
        return 0

    from .catalog import Catalog
    from .config import OpenSubtitlesSettings, get_paths
    from .opensubtitles import OpenSubtitlesClient

    paths = get_paths()
    catalog = Catalog(paths.db)
    client = OpenSubtitlesClient(OpenSubtitlesSettings.from_env())

    if args.cmd == "scan":
        from .listing import build_listing

        build_listing(client, catalog, args.src, args.tgt, _log, refresh=args.refresh)
    elif args.cmd == "prepare":
        from .prepare import prepare

        feature_ids = json.loads(Path(args.features_file).read_text())
        prepare(client, catalog, paths, args.src, args.tgt, feature_ids, _log)
    elif args.cmd == "train":
        from .runconfig import TrainConfig
        from .runner import run_job

        run_job(TrainConfig.load(Path(args.config)), _log)
    elif args.cmd == "quota":
        q = client.quota()
        print(json.dumps({"allowed": q.allowed, "remaining": q.remaining, "vip": q.vip}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
