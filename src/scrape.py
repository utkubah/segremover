"""
Collect YouTube transcripts for the corpus.

Usage:
    # Run all sources defined in sources.csv
    python src/scrape.py --sources sources.csv

    # Authenticate via browser cookies (recommended — avoids IP blocks)
    python src/scrape.py --sources sources.csv --cookies-from-browser chrome

    # Run a single playlist / channel
    python src/scrape.py --playlist URL --genre lectures --target 1250

    # Dry-run on 5 videos to test the pipeline
    python src/scrape.py --playlist URL --genre lectures --target 5 --dry-run

Output:
    data/raw/<genre>/<video_id>.json   one file per collected video
    data/scrape_checkpoint.jsonl       progress log (enables resume after crash)
"""

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import yt_dlp
from tqdm import tqdm

RAW_DIR = Path("data/raw")
CHECKPOINT = Path("data/scrape_checkpoint.jsonl")
MIN_WORDS = 500    # drop transcripts shorter than this
SLEEP_SEC = 2.0    # polite rate limit between transcript requests


# ── Helpers ────────────────────────────────────────────────────────────────────

def length_bucket(duration_sec) -> str:
    if duration_sec is None:
        return "unknown"
    if duration_sec < 300:
        return "short"
    if duration_sec < 1200:
        return "medium"
    return "long"


def load_done() -> set[str]:
    if not CHECKPOINT.exists():
        return set()
    return {json.loads(line)["video_id"] for line in CHECKPOINT.open(encoding="utf-8")}


def mark_done(video_id: str, status: str) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"video_id": video_id, "status": status}) + "\n")


# ── Fetching ───────────────────────────────────────────────────────────────────

def _apply_cookies(opts: dict, cookies_from_browser: str | None, cookiefile: str | None) -> None:
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    elif cookiefile:
        opts["cookiefile"] = cookiefile


def fetch_playlist(url: str, limit: int, cookies_from_browser: str | None = None, cookiefile: str | None = None) -> list[dict]:
    """Return video metadata entries from a playlist or channel URL."""
    opts = {
        "quiet": True,
        "extract_flat": True,
        "playlistend": limit * 3,
    }
    _apply_cookies(opts, cookies_from_browser, cookiefile)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries") or [info]
    return [
        {
            "video_id": e["id"],
            "title": e.get("title"),
            "channel": e.get("channel") or info.get("channel"),
            "channel_id": e.get("channel_id") or info.get("channel_id"),
            "duration": e.get("duration"),
            "upload_date": e.get("upload_date"),
        }
        for e in entries
        if e.get("id")
    ]


def _parse_json3(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for event in data.get("events", []):
        if "segs" not in event:
            continue
        text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
        if not text or text == "\n":
            continue
        entries.append({
            "text": text,
            "start": event["tStartMs"] / 1000.0,
            "duration": event.get("dDurationMs", 0) / 1000.0,
        })
    return entries


def fetch_transcript(video_id: str, cookies_from_browser: str | None = None, cookiefile: str | None = None) -> tuple[list[dict], str]:
    """
    Fetch transcript using yt-dlp. Prefers manual English captions over auto-generated.
    Returns (entries, caption_type) where entries is a list of
    {"text": str, "start": float, "duration": float} dicts.
    Raises FileNotFoundError if no English transcript is available.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "json3",
            "outtmpl": os.path.join(tmpdir, "%(id)s"),
        }
        _apply_cookies(opts, cookies_from_browser, cookiefile)

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)

        sub_path = os.path.join(tmpdir, f"{video_id}.en.json3")
        if not os.path.exists(sub_path):
            raise FileNotFoundError(f"No English transcript for {video_id}")

        has_manual = bool(info.get("subtitles", {}).get("en"))
        caption_type = "manual" if has_manual else "auto"

        return _parse_json3(sub_path), caption_type


# ── Per-video processing ───────────────────────────────────────────────────────

def process(meta: dict, genre: str, dry_run: bool = False, cookies_from_browser: str | None = None, cookiefile: str | None = None) -> str:
    """
    Fetch and save one video. Returns status string.
    Statuses: ok | no_transcript | too_short | error
    """
    vid = meta["video_id"]
    out = RAW_DIR / genre / f"{vid}.json"

    if dry_run:
        print(f"  [dry-run] would process {vid} — {meta.get('title', '')[:60]}")
        return "ok"

    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        entries, caption_type = fetch_transcript(vid, cookies_from_browser=cookies_from_browser, cookiefile=cookiefile)
    except FileNotFoundError:
        return "no_transcript"
    except Exception as exc:
        print(f"  [error] {vid}: {type(exc).__name__}: {exc}")
        return "error"

    word_count = sum(len(e["text"].split()) for e in entries)
    if word_count < MIN_WORDS:
        return "too_short"

    doc = {
        "video_id": vid,
        "genre": genre,
        "caption_type": caption_type,
        "length_bucket": length_bucket(meta.get("duration")),
        **meta,
        "word_count": word_count,
        "transcript": entries,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return "ok"


# ── Main run loop ──────────────────────────────────────────────────────────────

def run(url: str, genre: str, target: int, dry_run: bool = False, cookies_from_browser: str | None = None, cookiefile: str | None = None) -> None:
    done = load_done()

    if dry_run:
        print("\n*** DRY RUN — no transcripts will be downloaded, no files will be saved ***")
    print(f"\n[{genre}] fetching playlist metadata from:\n  {url}")
    candidates = fetch_playlist(url, target, cookies_from_browser=cookies_from_browser, cookiefile=cookiefile)
    candidates = [c for c in candidates if c["video_id"] not in done]
    print(f"[{genre}] {len(candidates)} candidates after skipping already-done")

    counts = {"ok": 0, "no_transcript": 0, "too_short": 0, "error": 0}

    with tqdm(total=target, desc=genre, unit="vid") as bar:
        for meta in candidates:
            if counts["ok"] >= target:
                break

            status = process(meta, genre, dry_run=dry_run, cookies_from_browser=cookies_from_browser, cookiefile=cookiefile)

            if not dry_run:
                mark_done(meta["video_id"], status)

            counts[status] += 1
            if status == "ok":
                bar.update(1)

            time.sleep(SLEEP_SEC)

    if dry_run:
        print(f"[{genre}] DRY RUN finished: {counts} — NO FILES SAVED. Re-run without --dry-run to actually download.")
    else:
        print(f"[{genre}] finished: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape YouTube transcripts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sources", help="CSV file with columns: url,genre,target")
    group.add_argument("--playlist", help="Single playlist or channel URL")
    parser.add_argument("--genre", help="Genre label (required with --playlist)")
    parser.add_argument("--target", type=int, default=1250, help="Max videos per source")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch playlist metadata only, do not download transcripts")
    parser.add_argument("--cookies-from-browser",
                        metavar="BROWSER",
                        help="Read cookies from a closed browser (e.g. chrome, firefox, edge)")
    parser.add_argument("--cookies",
                        metavar="FILE",
                        help="Path to a Netscape-format cookies.txt file")
    args = parser.parse_args()

    cfb = getattr(args, "cookies_from_browser", None)
    cookiefile = args.cookies or None
    if cookiefile and not Path(cookiefile).exists():
        parser.error(f"Cookies file not found: {cookiefile}")

    if args.sources:
        with open(args.sources, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["url"].startswith("TBD"):
                    print(f"[skip] {row['genre']} — URL not yet set in sources.csv")
                    continue
                run(row["url"], row["genre"], int(row["target"]), dry_run=args.dry_run, cookies_from_browser=cfb, cookiefile=cookiefile)
    else:
        if not args.genre:
            parser.error("--genre is required when using --playlist")
        run(args.playlist, args.genre, args.target, dry_run=args.dry_run, cookies_from_browser=cfb, cookiefile=cookiefile)


if __name__ == "__main__":
    main()
