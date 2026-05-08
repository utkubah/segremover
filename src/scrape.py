"""
Collect a diverse YouTube transcript corpus for segremover.

This scraper reads a source registry CSV and writes one JSON document per video.
The registry is intentionally organized around six broad genres:

- lectures
- commentary
- ted
- tv_series
- podcasts
- entertainment

It supports channel, playlist, URL, and YouTube search sources, applies per-genre
quotas, caps successful videos per channel, logs every candidate, supports
local/HPC sharding, and enforces English transcripts/subtitles for every saved
example. Non-English sources such as TV dramas are allowed only when English
subtitles are available.

Examples
--------
Local pilot dry run:
    python src/scrape.py --sources data/source_registry.csv --target-total 100 --dry-run

Local pilot scrape:
    python src/scrape.py --sources data/source_registry.csv --target-total 300 \
      --cookies-from-browser chrome --candidate-log data/candidates_pilot.csv

Full scrape:
    python src/scrape.py --sources data/source_registry.csv --target-total 5000 \
      --cookies-from-browser chrome --candidate-log data/candidates_full.csv

HPC shard:
    python src/scrape.py --sources data/source_registry.csv --target-total 5000 \
      --shard-id ${SLURM_ARRAY_TASK_ID} --num-shards ${SLURM_ARRAY_TASK_COUNT} \
      --candidate-log data/candidates_shard_${SLURM_ARRAY_TASK_ID}.csv

Output
------
    data/raw/<genre>/<video_id>.json
    data/scrape_checkpoint.jsonl
    data/candidates*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yt_dlp
from tqdm import tqdm

RAW_DIR = Path("data/raw")
CHECKPOINT = Path("data/scrape_checkpoint.jsonl")
DEFAULT_CANDIDATE_LOG = Path("data/candidates.csv")

MIN_WORDS_DEFAULT = 500
SLEEP_SEC_DEFAULT = 2.0
MAX_CANDIDATE_MULTIPLIER = 5

GENRE_TARGETS_DEFAULT = {
    "lectures": 1200,
    "commentary": 900,
    "ted": 700,
    "tv_series": 480,
    "podcasts": 900,
    "entertainment": 820,
}

ALLOWED_GENRES = set(GENRE_TARGETS_DEFAULT)
ALLOWED_SOURCE_TYPES = {"channel", "playlist", "url", "search"}

CANDIDATE_COLUMNS = [
    "video_id",
    "status",
    "genre",
    "source_id",
    "source_type",
    "source_value",
    "title",
    "channel",
    "channel_id",
    "duration",
    "length_bucket",
    "upload_date",
    "language_hint",
    "caption_language",
    "caption_type",
    "english_subtitles_available",
    "word_count",
    "view_count",
    "like_count",
    "reason",
    "url",
]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    source_type: str
    source_value: str
    genre: str
    target: int
    language_hint: str = "en"
    requires_english_subs: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def parse_bool(value: str | bool | None, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def length_bucket(duration_sec: int | float | None) -> str:
    if duration_sec is None:
        return "unknown"
    try:
        duration = float(duration_sec)
    except (TypeError, ValueError):
        return "unknown"
    if duration < 300:
        return "short"
    if duration < 1200:
        return "medium"
    return "long"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def count_words(entries: list[dict[str, Any]]) -> int:
    return sum(len(clean_text(e.get("text", "")).split()) for e in entries)


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _apply_cookies(opts: dict[str, Any], cookies_from_browser: str | None, cookiefile: str | None) -> None:
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    elif cookiefile:
        opts["cookiefile"] = cookiefile


def load_done(checkpoint: Path = CHECKPOINT) -> set[str]:
    if not checkpoint.exists():
        return set()
    done: set[str] = set()
    with checkpoint.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("video_id"):
                done.add(row["video_id"])
    return done


def mark_done(video_id: str, status: str, checkpoint: Path = CHECKPOINT, extra: dict[str, Any] | None = None) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    row = {"video_id": video_id, "status": status, "time": time.time()}
    if extra:
        row.update(extra)
    with checkpoint.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_candidate_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    normalized = {col: row.get(col, "") for col in CANDIDATE_COLUMNS}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


# ---------------------------------------------------------------------------
# Registry loading and quotas
# ---------------------------------------------------------------------------


def load_sources(path: Path) -> list[SourceSpec]:
    rows: list[SourceSpec] = []
    seen_source_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"source_id", "source_type", "source_value", "genre", "target"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"source registry missing columns: {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            if not raw.get("source_value") or raw["source_value"].startswith("TBD"):
                continue

            source_id = raw["source_id"].strip()
            source_type = raw["source_type"].strip().lower()
            source_value = raw["source_value"].strip()
            genre = raw["genre"].strip()

            if not source_id:
                raise ValueError(f"source registry line {line_number}: empty source_id")
            if source_id in seen_source_ids:
                raise ValueError(f"source registry line {line_number}: duplicate source_id={source_id!r}")
            seen_source_ids.add(source_id)

            if source_type not in ALLOWED_SOURCE_TYPES:
                raise ValueError(
                    f"source registry line {line_number}: source_type={source_type!r}; "
                    f"expected one of {sorted(ALLOWED_SOURCE_TYPES)}"
                )
            if genre not in ALLOWED_GENRES:
                raise ValueError(
                    f"source registry line {line_number}: genre={genre!r}; "
                    f"expected one of {sorted(ALLOWED_GENRES)}"
                )

            try:
                target = int(raw.get("target") or 0)
            except ValueError as exc:
                raise ValueError(f"source registry line {line_number}: target must be an integer") from exc
            if target < 0:
                raise ValueError(f"source registry line {line_number}: target must be >= 0")

            rows.append(
                SourceSpec(
                    source_id=source_id,
                    source_type=source_type,
                    source_value=source_value,
                    genre=genre,
                    target=target,
                    language_hint=(raw.get("language_hint") or "en").strip(),
                    requires_english_subs=parse_bool(raw.get("requires_english_subs"), default=True),
                    notes=(raw.get("notes") or "").strip(),
                )
            )
    return rows


def load_genre_targets(json_arg: str | None) -> dict[str, int]:
    if not json_arg:
        return dict(GENRE_TARGETS_DEFAULT)
    try:
        data = json.loads(json_arg)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--genre-targets-json must be valid JSON: {exc}") from exc
    return {str(k): int(v) for k, v in data.items()}


def shard_sources(sources: list[SourceSpec], shard_id: int | None, num_shards: int | None) -> list[SourceSpec]:
    if shard_id is None and num_shards is None:
        return sources
    if shard_id is None or num_shards is None:
        raise ValueError("--shard-id and --num-shards must be provided together")
    if num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    return [src for idx, src in enumerate(sources) if idx % num_shards == shard_id]


# ---------------------------------------------------------------------------
# Metadata discovery
# ---------------------------------------------------------------------------


def ydl_base_opts(cookies_from_browser: str | None, cookiefile: str | None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "retries": 3,
        "extractor_retries": 3,
    }
    _apply_cookies(opts, cookies_from_browser, cookiefile)
    return opts


def discovery_url(source: SourceSpec, max_candidates: int) -> str:
    if source.source_type in {"channel", "playlist", "url"}:
        return source.source_value
    if source.source_type == "search":
        safe_n = max(1, max_candidates)
        return f"ytsearch{safe_n}:{source.source_value}"
    raise ValueError(f"unknown source_type={source.source_type!r}; expected channel, playlist, url, or search")


def normalize_flat_entry(entry: dict[str, Any], parent_info: dict[str, Any], source: SourceSpec) -> dict[str, Any] | None:
    video_id = entry.get("id")
    if not video_id:
        url = entry.get("url") or entry.get("webpage_url")
        if url and "watch?v=" in url:
            video_id = url.split("watch?v=", 1)[1].split("&", 1)[0]
    if not video_id:
        return None
    return {
        "video_id": video_id,
        "title": entry.get("title"),
        "channel": entry.get("channel") or entry.get("uploader") or parent_info.get("channel") or parent_info.get("uploader"),
        "channel_id": entry.get("channel_id") or entry.get("uploader_id") or parent_info.get("channel_id") or parent_info.get("uploader_id"),
        "duration": entry.get("duration"),
        "upload_date": entry.get("upload_date"),
        "view_count": entry.get("view_count"),
        "like_count": entry.get("like_count"),
        "source_id": source.source_id,
        "source_type": source.source_type,
        "source_value": source.source_value,
        "genre": source.genre,
        "language_hint": source.language_hint,
        "requires_english_subs": source.requires_english_subs,
    }


def fetch_candidates(
    source: SourceSpec,
    cookies_from_browser: str | None,
    cookiefile: str | None,
    candidate_multiplier: int = MAX_CANDIDATE_MULTIPLIER,
) -> list[dict[str, Any]]:
    """Return flat metadata candidates from a channel, playlist, URL, or search query."""
    max_candidates = max(source.target * candidate_multiplier, source.target + 25)
    url = discovery_url(source, max_candidates=max_candidates)
    opts = ydl_base_opts(cookies_from_browser, cookiefile)
    opts.update(
        {
            "extract_flat": "in_playlist",
            "playlistend": max_candidates,
            "skip_download": True,
        }
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        return []
    entries = info.get("entries") or [info]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry:
            continue
        normalized = normalize_flat_entry(entry, info, source)
        if not normalized:
            continue
        vid = normalized["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        out.append(normalized)
    return out


# ---------------------------------------------------------------------------
# Transcript fetching
# ---------------------------------------------------------------------------


def _parse_json3(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries: list[dict[str, Any]] = []
    for event in data.get("events", []):
        if "segs" not in event:
            continue
        text = clean_text("".join(seg.get("utf8", "") for seg in event["segs"]))
        if not text:
            continue
        entries.append(
            {
                "text": text,
                "start": event.get("tStartMs", 0) / 1000.0,
                "duration": event.get("dDurationMs", 0) / 1000.0,
            }
        )
    return entries


def _caption_lang_available(info: dict[str, Any], lang_prefix: str = "en") -> tuple[bool, str | None, str | None]:
    subtitles = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    manual_langs = sorted([lang for lang in subtitles if lang == lang_prefix or lang.startswith(f"{lang_prefix}-")])
    auto_langs = sorted([lang for lang in auto if lang == lang_prefix or lang.startswith(f"{lang_prefix}-")])

    if manual_langs:
        return True, manual_langs[0], "manual"
    if auto_langs:
        return True, auto_langs[0], "auto"
    return False, None, None


def fetch_transcript(
    video_id: str,
    cookies_from_browser: str | None = None,
    cookiefile: str | None = None,
    subtitle_lang: str = "en",
    allow_auto: bool = True,
) -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
    """
    Fetch English transcript with yt-dlp.

    Returns (entries, caption_type, caption_language, full_info).
    Raises FileNotFoundError if no usable English transcript is available.
    """
    url = video_url(video_id)
    with tempfile.TemporaryDirectory() as tmpdir:
        opts = ydl_base_opts(cookies_from_browser, cookiefile)
        opts.update(
            {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": allow_auto,
                "subtitleslangs": [subtitle_lang, f"{subtitle_lang}.*"],
                "subtitlesformat": "json3",
                "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            }
        )
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)

        if not info:
            raise FileNotFoundError(f"No metadata returned for {video_id}")

        has_caption, caption_language, caption_type = _caption_lang_available(info, subtitle_lang)
        if not has_caption or not caption_language or not caption_type:
            raise FileNotFoundError(f"No {subtitle_lang} transcript listed for {video_id}")

        # yt-dlp usually writes <id>.<lang>.json3. Be permissive because language
        # variants can appear as en-US, en-orig, etc.
        patterns = [
            os.path.join(tmpdir, f"{video_id}.{caption_language}.json3"),
            os.path.join(tmpdir, f"{video_id}.{subtitle_lang}.json3"),
            os.path.join(tmpdir, f"{video_id}.{subtitle_lang}-*.json3"),
            os.path.join(tmpdir, f"{video_id}.*.json3"),
        ]
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern))
        matches = sorted(set(matches))
        if not matches:
            raise FileNotFoundError(f"No downloaded {subtitle_lang} json3 subtitle file for {video_id}")

        entries = _parse_json3(matches[0])
        if not entries:
            raise FileNotFoundError(f"Empty transcript for {video_id}")
        return entries, caption_type, caption_language, info


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------


def merge_meta(flat_meta: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Prefer detailed yt-dlp metadata, fall back to flat discovery metadata."""
    return {
        "video_id": flat_meta["video_id"],
        "title": info.get("title") or flat_meta.get("title"),
        "channel": info.get("channel") or info.get("uploader") or flat_meta.get("channel"),
        "channel_id": info.get("channel_id") or info.get("uploader_id") or flat_meta.get("channel_id"),
        "duration": info.get("duration") or flat_meta.get("duration"),
        "upload_date": info.get("upload_date") or flat_meta.get("upload_date"),
        "view_count": info.get("view_count") or flat_meta.get("view_count"),
        "like_count": info.get("like_count") or flat_meta.get("like_count"),
        "webpage_url": info.get("webpage_url") or video_url(flat_meta["video_id"]),
        "description": info.get("description"),
        "tags": info.get("tags"),
    }


def candidate_row(
    meta: dict[str, Any],
    status: str,
    reason: str = "",
    caption_language: str = "",
    caption_type: str = "",
    word_count: int | str = "",
) -> dict[str, Any]:
    duration = meta.get("duration")
    return {
        "video_id": meta.get("video_id", ""),
        "status": status,
        "genre": meta.get("genre", ""),
        "source_id": meta.get("source_id", ""),
        "source_type": meta.get("source_type", ""),
        "source_value": meta.get("source_value", ""),
        "title": meta.get("title", ""),
        "channel": meta.get("channel", ""),
        "channel_id": meta.get("channel_id", ""),
        "duration": duration or "",
        "length_bucket": length_bucket(duration),
        "upload_date": meta.get("upload_date", ""),
        "language_hint": meta.get("language_hint", ""),
        "caption_language": caption_language,
        "caption_type": caption_type,
        "english_subtitles_available": "yes" if caption_language.startswith("en") else "no",
        "word_count": word_count,
        "view_count": meta.get("view_count", ""),
        "like_count": meta.get("like_count", ""),
        "reason": reason,
        "url": meta.get("webpage_url") or video_url(str(meta.get("video_id", ""))),
    }


def process_video(
    flat_meta: dict[str, Any],
    out_dir: Path,
    min_words: int,
    dry_run: bool,
    cookies_from_browser: str | None,
    cookiefile: str | None,
    allow_auto: bool,
    candidate_log: Path,
    checkpoint: Path,
) -> str:
    vid = flat_meta["video_id"]
    genre = flat_meta["genre"]
    out = out_dir / genre / f"{vid}.json"

    if dry_run:
        append_candidate_log(candidate_log, candidate_row(flat_meta, "dry_run", reason="metadata only"))
        return "dry_run"

    if out.exists():
        append_candidate_log(candidate_log, candidate_row(flat_meta, "already_exists", reason=str(out)))
        return "already_exists"

    try:
        entries, caption_type, caption_language, info = fetch_transcript(
            vid,
            cookies_from_browser=cookies_from_browser,
            cookiefile=cookiefile,
            subtitle_lang="en",
            allow_auto=allow_auto,
        )
    except FileNotFoundError as exc:
        append_candidate_log(candidate_log, candidate_row(flat_meta, "no_english_transcript", reason=str(exc)))
        mark_done(vid, "no_english_transcript", checkpoint, {"genre": genre})
        return "no_english_transcript"
    except Exception as exc:  # keep scraping instead of dying on one video
        append_candidate_log(candidate_log, candidate_row(flat_meta, "error", reason=f"{type(exc).__name__}: {exc}"))
        mark_done(vid, "error", checkpoint, {"genre": genre})
        return "error"

    full_meta = merge_meta(flat_meta, info)
    full_meta.update(
        {
            "genre": genre,
            "source_id": flat_meta.get("source_id"),
            "source_type": flat_meta.get("source_type"),
            "source_value": flat_meta.get("source_value"),
            "language_hint": flat_meta.get("language_hint"),
            "requires_english_subs": flat_meta.get("requires_english_subs", True),
        }
    )
    wc = count_words(entries)
    if wc < min_words:
        append_candidate_log(
            candidate_log,
            candidate_row(full_meta, "too_short", reason=f"word_count<{min_words}", caption_language=caption_language, caption_type=caption_type, word_count=wc),
        )
        mark_done(vid, "too_short", checkpoint, {"genre": genre, "word_count": wc})
        return "too_short"

    doc = {
        "video_id": vid,
        "genre": genre,
        "source_id": flat_meta.get("source_id"),
        "source_type": flat_meta.get("source_type"),
        "source_value": flat_meta.get("source_value"),
        "language_hint": flat_meta.get("language_hint"),
        "requires_english_subs": flat_meta.get("requires_english_subs", True),
        "caption_language": caption_language,
        "caption_type": caption_type,
        "english_subtitles_available": True,
        "length_bucket": length_bucket(full_meta.get("duration")),
        "word_count": wc,
        **full_meta,
        "transcript": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    append_candidate_log(candidate_log, candidate_row(doc, "ok", caption_language=caption_language, caption_type=caption_type, word_count=wc))
    mark_done(vid, "ok", checkpoint, {"genre": genre, "word_count": wc})
    return "ok"


# ---------------------------------------------------------------------------
# Run loop with diversity constraints
# ---------------------------------------------------------------------------


def existing_success_counts(out_dir: Path) -> tuple[Counter[str], Counter[str], set[str]]:
    genre_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    done: set[str] = set()
    for path in out_dir.glob("*/*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        vid = doc.get("video_id")
        if vid:
            done.add(vid)
        genre = doc.get("genre") or path.parent.name
        genre_counts[genre] += 1
        channel_key = doc.get("channel_id") or doc.get("channel") or "unknown"
        channel_counts[channel_key] += 1
    return genre_counts, channel_counts, done


def run(args: argparse.Namespace) -> None:
    sources = load_sources(Path(args.sources))
    sources = shard_sources(sources, args.shard_id, args.num_shards)
    genre_targets = load_genre_targets(args.genre_targets_json)

    out_dir = Path(args.out_dir)
    checkpoint = Path(args.checkpoint)
    candidate_log = Path(args.candidate_log)

    checkpoint_done = load_done(checkpoint)
    genre_counts, channel_counts, existing_files = existing_success_counts(out_dir)
    done = checkpoint_done | existing_files

    global_seen: set[str] = set(done)
    total_ok_start = sum(genre_counts.values())
    total_target = int(args.target_total)

    print(f"Loaded {len(sources)} source rows")
    print(f"Existing OK files: {total_ok_start}")
    print(f"Target total: {total_target}")
    print(f"Candidate log: {candidate_log}")
    if args.dry_run:
        print("DRY RUN: metadata will be discovered but transcripts will not be downloaded")

    overall_counts: Counter[str] = Counter()

    for source in sources:
        if sum(genre_counts.values()) >= total_target:
            break
        if genre_counts[source.genre] >= genre_targets.get(source.genre, source.target):
            print(f"[skip] {source.source_id}: genre target already met for {source.genre}")
            continue

        remaining_genre = genre_targets.get(source.genre, source.target) - genre_counts[source.genre]
        remaining_total = total_target - sum(genre_counts.values())
        source_target = max(0, min(source.target, remaining_genre, remaining_total))
        if source_target <= 0:
            continue

        print(f"\n[{source.genre}] {source.source_id} ({source.source_type}) target={source_target}")
        try:
            candidates = fetch_candidates(
                source,
                cookies_from_browser=args.cookies_from_browser,
                cookiefile=args.cookies,
                candidate_multiplier=args.candidate_multiplier,
            )
        except Exception as exc:
            print(f"  [source_error] {source.source_id}: {type(exc).__name__}: {exc}")
            continue

        print(f"  discovered {len(candidates)} candidates")
        ok_from_source = 0
        source_counts: Counter[str] = Counter()

        with tqdm(total=source_target, desc=source.source_id, unit="vid") as bar:
            for meta in candidates:
                if ok_from_source >= source_target:
                    break
                if sum(genre_counts.values()) >= total_target:
                    break
                if genre_counts[source.genre] >= genre_targets.get(source.genre, source.target):
                    break

                vid = meta["video_id"]
                if vid in global_seen:
                    continue
                global_seen.add(vid)

                channel_key = meta.get("channel_id") or meta.get("channel") or "unknown"
                if channel_counts[channel_key] >= args.max_per_channel:
                    append_candidate_log(candidate_log, candidate_row(meta, "skipped_channel_cap", reason=f"channel_cap={args.max_per_channel}"))
                    source_counts["skipped_channel_cap"] += 1
                    continue

                status = process_video(
                    meta,
                    out_dir=out_dir,
                    min_words=args.min_words,
                    dry_run=args.dry_run,
                    cookies_from_browser=args.cookies_from_browser,
                    cookiefile=args.cookies,
                    allow_auto=not args.no_auto_subs,
                    candidate_log=candidate_log,
                    checkpoint=checkpoint,
                )
                source_counts[status] += 1
                overall_counts[status] += 1

                if status == "ok":
                    ok_from_source += 1
                    genre_counts[source.genre] += 1
                    channel_counts[channel_key] += 1
                    bar.update(1)
                elif status == "dry_run":
                    ok_from_source += 1
                    genre_counts[source.genre] += 1
                    bar.update(1)

                time.sleep(args.sleep_sec)

        print(f"  source counts: {dict(source_counts)}")
        print(f"  genre progress: {source.genre}={genre_counts[source.genre]}/{genre_targets.get(source.genre, source.target)}")

    print("\nDone.")
    print(f"Overall status counts: {dict(overall_counts)}")
    print(f"Genre counts: {dict(genre_counts)}")
    print("Top channels:")
    for channel, count in channel_counts.most_common(15):
        print(f"  {channel}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape a diverse YouTube transcript corpus")
    parser.add_argument("--sources", required=True, help="CSV source registry")
    parser.add_argument("--target-total", type=int, default=5000, help="Total successful videos to collect")
    parser.add_argument("--genre-targets-json", default=None, help="Optional JSON dict overriding default per-genre targets")
    parser.add_argument("--out-dir", default=str(RAW_DIR), help="Output root for data/raw JSON files")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT), help="Checkpoint JSONL path")
    parser.add_argument("--candidate-log", default=str(DEFAULT_CANDIDATE_LOG), help="Candidate CSV log path")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS_DEFAULT, help="Minimum transcript word count")
    parser.add_argument("--max-per-channel", type=int, default=200, help="Soft cap for successful videos per channel")
    parser.add_argument("--candidate-multiplier", type=int, default=MAX_CANDIDATE_MULTIPLIER, help="Discover this many times each source target")
    parser.add_argument("--sleep-sec", type=float, default=SLEEP_SEC_DEFAULT, help="Delay between transcript fetches")
    parser.add_argument("--dry-run", action="store_true", help="Discover metadata only; do not download transcripts")
    parser.add_argument("--no-auto-subs", action="store_true", help="Use manual subtitles only")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER", help="Read cookies from browser, e.g. chrome or firefox")
    parser.add_argument("--cookies", metavar="FILE", help="Path to Netscape-format cookies.txt")
    parser.add_argument("--shard-id", type=int, default=None, help="HPC array shard id, 0-indexed")
    parser.add_argument("--num-shards", type=int, default=None, help="HPC total number of shards")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cookies and not Path(args.cookies).exists():
        parser.error(f"cookies file not found: {args.cookies}")
    run(args)


if __name__ == "__main__":
    main()
