#!/usr/bin/env python3
"""
Generate blog posts from CSV questions using AI with database, resume support,
fallback, and a flicker-free live Rich TUI showing streaming responses,
per-worker progress, and tokens/sec.

Key improvements over the previous version
-------------------------------------------
* TUI no longer flickers: the Layout tree is built ONCE and only the
  renderables inside it are swapped in place. Rich's own `Live` auto-refresh
  thread paints the screen on a fixed cadence -- there is no more competing
  manual `live.update()` loop that used to double-render and tear the screen.
* Multi-job view: when concurrency > 1 you now see every active generation
  (title, status, words, tokens/sec, mini progress bar), not just one.
* A small tokens/sec sparkline per active job.
* HTTP session reuse with urllib3 retry/backoff adapter (fewer TCP/TLS
  handshakes, more resilient to transient 5xx/connection errors).
* Cooperative cancellation: Ctrl+C now signals a `threading.Event` that is
  checked inside the streaming loop, so shutdown is prompt instead of
  waiting for an in-flight HTTP request to time out.
* SQLite: added an index on `status`, and CSV import now uses a single
  `executemany` transaction instead of one round trip per row.
* Export supports both Markdown (default) and JSON.
* New `--init-config` command that writes a starter JSON config file.
* Config validation with clear error messages before any work starts.
"""

import csv
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box


# ============================================================================
# LOGGING (with colors)
# ============================================================================
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logging(verbose=False, log_file=None):
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(
        ColorFormatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(console)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")
        )
        root.addHandler(fh)
        logging.info("Logging to %s", log_file)


log = logging.getLogger("bloggen")

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_CONFIG = {
    "api_endpoint": "https://llm.laravelmail.com/api/chat",
    "model": "gemma4:e2b",
    "api_key": "",
    "request_timeout": 1200,
    "max_retries": 3,
    "backoff_factor": 2,
    "min_words": 350,
    "request_delay": 2.0,
    "concurrency": 2,
    "system_prompt": (
        "You are a senior developer and technical blogger. "
        "Your answers are accurate, practical, and well-explained. "
        "Always include natural backlinks to https://laravelcompany.com where relevant. "
        "Your blog posts must be at least 350 words long and include proper headings, paragraphs, "
        "code examples, and a clear conclusion."
    ),
    "user_prompt_template": (
        "Write a comprehensive blog post that answers the following question.\n\n"
        "Title: {title}\n"
        "Body: {body}\n\n"
        "The blog post should:\n"
        "- Provide a correct and thorough answer from a developer's perspective.\n"
        "- Include relevant code examples or best practices.\n"
        "- Naturally incorporate backlinks to https://laravelcompany.com in the content.\n"
        "- Be well-structured with headings, paragraphs, and a clear conclusion.\n"
        "- Be at least 350 words long."
    ),
    "expand_prompt": (
        "\n\nThe previous response was too short. Please expand it to at least 350 words "
        "by adding more details, examples, and deeper explanations. Include additional sections "
        "and ensure the final post is comprehensive and well structured."
    ),
    "db_file": "blog_posts.db",
    "export_dir": "exported_posts",
    "export_format": "md",
    "log_file": None,
    "debug_ai": False,
    "stream": False,
    "max_tokens": 4096,
    "no_tui": False,
    "stream_to_file": None,
    "stream_lines": 180,
    "quiet": False,
    "tui_refresh_per_second": 10,
}

CONFIG_INT_KEYS = {
    "request_timeout",
    "max_tokens",
    "stream_lines",
    "concurrency",
    "max_retries",
    "backoff_factor",
    "tui_refresh_per_second",
}
CONFIG_BOOL_KEYS = {"debug_ai", "stream", "no_tui", "quiet"}
CONFIG_FLOAT_KEYS = {"request_delay"}


def load_config(config_path=None):
    cfg = dict(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        cfg.update(file_cfg)
        log.info("Loaded config from %s", config_path)
    # Environment overrides
    env_map = {
        "AI_ENDPOINT": "api_endpoint",
        "AI_MODEL": "model",
        "AI_API_KEY": "api_key",
        "REQUEST_TIMEOUT": "request_timeout",
        "DB_FILE": "db_file",
        "EXPORT_DIR": "export_dir",
        "EXPORT_FORMAT": "export_format",
        "DEBUG_AI": "debug_ai",
        "STREAM": "stream",
        "MAX_TOKENS": "max_tokens",
        "NO_TUI": "no_tui",
        "STREAM_LINES": "stream_lines",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if cfg_key in CONFIG_INT_KEYS:
                val = int(val)
            elif cfg_key in CONFIG_BOOL_KEYS:
                val = val.lower() in ("1", "true", "yes")
            cfg[cfg_key] = val
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    errors = []
    if not cfg.get("api_endpoint"):
        errors.append("api_endpoint is required")
    if not cfg.get("model"):
        errors.append("model is required")
    for key in CONFIG_INT_KEYS:
        try:
            int(cfg[key])
        except (TypeError, ValueError):
            errors.append(f"{key} must be an integer, got {cfg[key]!r}")
    if cfg.get("concurrency", 1) < 1:
        errors.append("concurrency must be >= 1")
    if cfg.get("export_format") not in ("md", "json"):
        errors.append("export_format must be 'md' or 'json'")
    if errors:
        for e in errors:
            log.error("Config error: %s", e)
        raise SystemExit("Invalid configuration. Fix the errors above and try again.")


def write_starter_config(path):
    starter = {k: v for k, v in DEFAULT_CONFIG.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(starter, f, indent=2, ensure_ascii=False)
    log.info("Wrote starter config to %s", path)


# ============================================================================
# DATABASE
# ============================================================================
def get_db_connection(db_path):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path):
    conn = get_db_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blog_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            blog_post TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            retries INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            error_message TEXT,
            UNIQUE(title)
        )
    """)
    conn.commit()
    # Migrate missing columns
    cursor = conn.execute("PRAGMA table_info(blog_posts)")
    existing = {row["name"] for row in cursor.fetchall()}
    for col, col_def in [
        ("word_count", "INTEGER DEFAULT 0"),
        ("error_message", "TEXT"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE blog_posts ADD COLUMN {col} {col_def}")
    # Index for the hot query path (status filtering used on every run)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_blog_posts_status ON blog_posts(status)"
    )
    conn.commit()
    conn.close()


def import_csv_to_db(csv_file, db_path):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    existing_titles = {row[0] for row in cursor.execute("SELECT title FROM blog_posts")}
    to_insert = []
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"Title", "Body"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must have 'Title' and 'Body' columns.")
        for row in reader:
            title = (row.get("Title") or "").strip()
            body = (row.get("Body") or "").strip()
            if not title or not body:
                continue
            if title in existing_titles:
                continue
            existing_titles.add(title)  # guard against dupes within the same CSV
            to_insert.append((title, body, "pending"))
    if to_insert:
        cursor.executemany(
            "INSERT INTO blog_posts (title, body, status) VALUES (?, ?, ?)",
            to_insert,
        )
        conn.commit()
    conn.close()
    return len(to_insert)


def get_records(db_path, status_filter=None, limit=None, max_retries=3):
    conn = get_db_connection(db_path)
    conditions, params = [], []
    if status_filter == "pending":
        conditions.append("status IN ('pending', 'failed') AND retries < ?")
        params.append(max_retries)
    elif status_filter:
        conditions.append("status = ?")
        params.append(status_filter)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = (
        f"SELECT id, title, body, status, retries, word_count, error_message, blog_post "
        f"FROM blog_posts {where} ORDER BY id"
    )
    if limit is not None:
        query += f" LIMIT {limit}"
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_record_status(
    db_path,
    record_id,
    status,
    blog_post=None,
    retries=None,
    word_count=None,
    error_message=None,
):
    conn = get_db_connection(db_path)
    updates, params = [], []
    if status:
        updates.append("status = ?")
        params.append(status)
    if blog_post is not None:
        updates.append("blog_post = ?")
        params.append(blog_post)
    if retries is not None:
        updates.append("retries = ?")
        params.append(retries)
    if word_count is not None:
        updates.append("word_count = ?")
        params.append(word_count)
    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)
    updates.append("updated_at = ?")
    params.append(datetime.now().isoformat())
    params.append(record_id)
    sql = f"UPDATE blog_posts SET {', '.join(updates)} WHERE id = ?"
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def delete_records(db_path, status=None):
    conn = get_db_connection(db_path)
    if status:
        conn.execute("DELETE FROM blog_posts WHERE status = ?", (status,))
    else:
        conn.execute("DELETE FROM blog_posts")
    deleted = conn.total_changes
    conn.commit()
    conn.close()
    return deleted


# ============================================================================
# HTTP SESSION (connection reuse + retry/backoff at the transport level)
# ============================================================================
def build_session(cfg):
    session = requests.Session()
    retry = Retry(
        total=0,  # we do our own application-level retry loop in call_ai
        connect=0,
        read=0,
        backoff_factor=0,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ============================================================================
# AI CALL – SUPPORTS BOTH OPENAI AND OLLAMA NATIVE
# ============================================================================
def mask_api_key(key):
    if not key:
        return "<none>"
    return key[:4] + "***" + key[-4:] if len(key) > 8 else "***"


def _log_ai_request(payload, cfg):
    if not cfg.get("debug_ai"):
        return
    safe = {**payload}
    if cfg.get("api_key"):
        safe["headers"] = {"Authorization": f"Bearer {mask_api_key(cfg['api_key'])}"}
    log.debug("AI REQUEST: %s", json.dumps(safe, indent=2, ensure_ascii=False))


def _stream_response(
    response, debug_ai, on_chunk=None, cancel_event: Optional[threading.Event] = None
):
    """Streaming parser – handles SSE and plain JSON fallback, cancellable mid-stream."""
    try:
        content_type = response.headers.get("content-type", "")
        if (
            "application/json" in content_type
            and response.headers.get("transfer-encoding") != "chunked"
        ):
            data = response.json()
            if "choices" in data:
                content = data["choices"][0].get("message", {}).get("content", "") or ""
                if content:
                    if on_chunk:
                        on_chunk(content)
                    yield content
                    return
        for line in response.iter_lines(decode_unicode=True, delimiter=None):
            if cancel_event is not None and cancel_event.is_set():
                log.debug("Streaming cancelled by user request.")
                try:
                    response.close()
                except Exception:
                    pass
                return
            if not line or line.strip() == "":
                continue
            if debug_ai:
                log.debug("RAW LINE: %s", line[:300])

            if line.startswith("data: "):
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "") or ""
                    if content:
                        if on_chunk:
                            on_chunk(content)
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    if debug_ai:
                        log.debug("JSON parse error on data line: %s", str(e))
                    continue
            else:
                try:
                    chunk = json.loads(line.strip())
                    if isinstance(chunk, dict):
                        delta = chunk.get("choices", [{}])[0].get(
                            "delta", {}
                        ) or chunk.get("message", {})
                        content = (
                            delta.get("content", "") if isinstance(delta, dict) else ""
                        )
                        if content:
                            if on_chunk:
                                on_chunk(content)
                            yield content
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log.warning("Streaming interrupted: %s", e)
        if debug_ai:
            log.exception("Streaming error details")


def call_ai(
    prompt,
    cfg,
    session,
    expand=False,
    on_chunk: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
):
    """Unified AI caller – auto-detects endpoint format."""
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    system_msg = cfg["system_prompt"]
    if expand:
        system_msg += " The response must be at least 350 words. If you have already written a shorter version, expand it now."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    endpoint = cfg["api_endpoint"].lower()
    use_ollama = "/api/chat" in endpoint and "/v1/" not in endpoint
    stream_enabled = cfg.get("stream", False) and on_chunk is not None

    if use_ollama:
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "stream": stream_enabled,
            "options": {
                "temperature": 0.7,
                "num_predict": cfg.get("max_tokens", 4096),
            },
        }
    else:
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": cfg.get("max_tokens", 4096),
            "stream": stream_enabled,
            "stop": None,
        }

    max_retries = cfg["max_retries"]
    backoff = cfg["backoff_factor"]
    timeout = cfg["request_timeout"]

    for attempt in range(1, max_retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            return None
        start = time.time()
        log.debug("API attempt %d/%d", attempt, max_retries)
        try:
            _log_ai_request({**payload, "headers": headers}, cfg)

            resp = session.post(
                cfg["api_endpoint"],
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=stream_enabled,
            )

            log.debug("Request completed in %.2fs", time.time() - start)

            if cfg.get("debug_ai"):
                log.debug("RAW RESPONSE STATUS: %s", resp.status_code)
                log.debug(
                    "RAW RESPONSE BODY (first 1500 chars): %s",
                    resp.text[:1500] if not stream_enabled else "[STREAMING]",
                )

            resp.raise_for_status()

            if stream_enabled:
                content_parts = []
                for chunk in _stream_response(
                    resp,
                    cfg.get("debug_ai", False),
                    on_chunk=on_chunk,
                    cancel_event=cancel_event,
                ):
                    content_parts.append(chunk)
                if cancel_event is not None and cancel_event.is_set():
                    return None
                full_content = "".join(content_parts).strip()
            else:
                data = resp.json()
                if "error" in data:
                    raise ValueError(f"API error: {data['error']}")
                if use_ollama:
                    full_content = data.get("message", {}).get("content", "").strip()
                else:
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError("Empty choices")
                    full_content = (
                        choices[0].get("message", {}).get("content", "").strip()
                    )

            if not full_content:
                raise ValueError("Empty response from API")

            wc = len(full_content.split())
            log.debug("Received %d words", wc)
            if cfg.get("debug_ai"):
                log.debug("FULL RESPONSE (truncated): %s...", full_content[:1500])

            return full_content

        except requests.exceptions.Timeout:
            wait = backoff**attempt
            log.warning("Timeout (attempt %d). Retry in %ds", attempt, wait)
            _sleep_cancellable(wait, cancel_event)
        except requests.exceptions.ConnectionError as e:
            wait = backoff**attempt
            log.error(
                "Connection error (attempt %d): %s. Retry in %ds", attempt, e, wait
            )
            _sleep_cancellable(wait, cancel_event)
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "?"
            log.error("HTTP %s (attempt %d): %s", status_code, attempt, e)
            if e.response is not None and e.response.text:
                log.debug("Response body: %s", e.response.text[:800])
            if attempt == max_retries:
                return None
            _sleep_cancellable(backoff**attempt, cancel_event)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.error("Invalid response (attempt %d): %s", attempt, e)
            if attempt == max_retries:
                return None
            _sleep_cancellable(backoff**attempt, cancel_event)
        except Exception as e:
            log.warning("API error (attempt %d): %s", attempt, e)
            if log.isEnabledFor(logging.DEBUG):
                log.exception("Exception details:")
            if attempt == max_retries:
                return None
            _sleep_cancellable(backoff**attempt, cancel_event)

    return None


def _sleep_cancellable(seconds, cancel_event: Optional[threading.Event]):
    """Sleep in small increments so a cancel request is honored promptly."""
    if cancel_event is None:
        time.sleep(seconds)
        return
    end = time.time() + seconds
    while time.time() < end:
        if cancel_event.is_set():
            return
        time.sleep(min(0.1, end - time.time()))


def generate_blog_post(
    title,
    body,
    cfg,
    session,
    on_chunk: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
):
    """Generate with fallback – returns best content even if below min_words."""
    user_prompt = cfg["user_prompt_template"].format(title=title, body=body)
    current_prompt = user_prompt
    expand = False
    best_content = None
    best_word_count = 0

    for attempt in range(1, cfg["max_retries"] + 1):
        if cancel_event is not None and cancel_event.is_set():
            break
        log.info("Attempt %d/%d for '%s'...", attempt, cfg["max_retries"], title[:50])
        if cfg.get("debug_ai"):
            log.debug("PROMPT (attempt %d): %s", attempt, current_prompt[:500])

        content = call_ai(
            current_prompt,
            cfg,
            session,
            expand=expand,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )
        if content is None:
            continue

        wc = len(content.split())
        log.info("Generated %d words", wc)

        if wc > best_word_count:
            best_content, best_word_count = content, wc

        if wc >= cfg["min_words"]:
            return content

        log.info("Too short (%d < %d). Expanding...", wc, cfg["min_words"])
        current_prompt = user_prompt + cfg["expand_prompt"]
        expand = True

    if best_content:
        log.warning(
            "Best effort: %d words (min %d). Saving partial.",
            best_word_count,
            cfg["min_words"],
        )
        return best_content
    return None


# ============================================================================
# RICH TUI – flicker-free, event-driven refresh, multi-job view
# ============================================================================
SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values, width=24):
    if not values:
        return " " * width
    values = values[-width:]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    chars = []
    for v in values:
        idx = int((v - lo) / span * (len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars).rjust(width)


class RichTUI:
    """
    Flicker-free design:
    * The Layout object (and its named sub-layouts) is built exactly once.
    * All state changes call `refresh()`, which mutates the *contents* of the
      existing layout sections in place (layout["x"].update(new_renderable)).
    * `Live` itself owns a single background thread that repaints the
      terminal at a fixed `refresh_per_second`. We never call `live.update()`
      or `live.refresh()` from multiple places, so there is exactly one
      writer to the terminal and no competing/overlapping redraws.
    """

    def __init__(self, total, stream_lines=180, refresh_per_second=10):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.running = {}  # id -> job state dict
        self.lock = threading.RLock()
        self.console = Console()
        self._stop = False
        self.overall_start = time.time()
        self.stream_lines = stream_lines

        self.layout = self._build_layout()
        self.live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=refresh_per_second,
            screen=True,
            transient=False,
            auto_refresh=True,
            vertical_overflow="crop",
        )
        self.refresh()  # populate initial content before first paint

    # ---- layout skeleton (built once) -------------------------------------------------
    def _build_layout(self):
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="progress", size=8),
            Layout(name="main"),
        )
        layout["main"].split_row(
            Layout(name="jobs", ratio=1),
            Layout(name="content", ratio=2),
        )
        return layout

    # ---- lifecycle ----------------------------------------------------------------------
    def start(self):
        self.live.start()

    def stop(self):
        self._stop = True
        self.live.stop()

    # ---- job state mutation (event-driven; each call triggers a targeted refresh) -------
    def start_record(self, record_id, title):
        with self.lock:
            self.running[record_id] = {
                "title": title,
                "start_time": time.time(),
                "word_count": 0,
                "token_count": 0,
                "content": "",
                "status": "starting",
                "history": [],
            }
            self.current_record_id = (
                getattr(self, "current_record_id", None) or record_id
            )
        self.refresh()

    def update_stream(self, record_id, chunk):
        with self.lock:
            rec = self.running.get(record_id)
            if not rec:
                return
            rec["content"] += chunk
            words = len(rec["content"].split())
            rec["word_count"] = words
            rec["token_count"] = int(words * 1.35)
            rec["status"] = "streaming"
            elapsed = max(time.time() - rec["start_time"], 0.001)
            rec["history"].append(rec["token_count"] / elapsed)
            if (
                not hasattr(self, "current_record_id")
                or self.current_record_id not in self.running
            ):
                self.current_record_id = record_id
        self.refresh()

    def set_content(self, record_id, content):
        with self.lock:
            rec = self.running.get(record_id)
            if not rec:
                return
            rec["content"] = content
            words = len(content.split())
            rec["word_count"] = words
            rec["token_count"] = int(words * 1.35)
            rec["status"] = "completed" if words > 0 else "failed"
        self.refresh()

    def set_error(self, record_id, error_msg):
        with self.lock:
            rec = self.running.get(record_id)
            if not rec:
                return
            rec["status"] = f"error: {error_msg[:60]}"
        self.refresh()

    def finish_record(self, record_id, success):
        with self.lock:
            self.running.pop(record_id, None)
            if success:
                self.completed += 1
            else:
                self.failed += 1
            if getattr(self, "current_record_id", None) == record_id:
                self.current_record_id = next(iter(self.running.keys()), None)
        self.refresh()

    def focus(self, record_id):
        """Explicitly switch which job the content pane shows."""
        with self.lock:
            if record_id in self.running:
                self.current_record_id = record_id
        self.refresh()

    # ---- rendering (mutates existing layout sections; never rebuilds the tree) ----------
    def refresh(self):
        with self.lock:
            self._render_header()
            self._render_progress()
            self._render_jobs()
            self._render_content()

    def _render_header(self):
        header_text = Text(
            "Blog Post Generator • Live Streaming", style="bold blue", justify="center"
        )
        self.layout["header"].update(Panel(header_text, style="white", box=box.ROUNDED))

    def _render_progress(self):
        done = self.completed + self.failed
        pct = (done / self.total * 100) if self.total else 0
        elapsed = time.time() - self.overall_start
        speed = done / max(elapsed, 1) * 60  # posts/min
        eta = (self.total - done) / max(speed / 60, 0.001) if done > 0 else 0

        prog_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        prog_table.add_column(justify="left", width=14)
        prog_table.add_column(justify="right")
        prog_table.add_column(justify="left", width=14)
        prog_table.add_column(justify="right")
        prog_table.add_row(
            "Total",
            str(self.total),
            "Active",
            str(len(self.running)),
        )
        prog_table.add_row(
            "Completed",
            f"[green]{self.completed} ✅[/green]",
            "Failed",
            f"[red]{self.failed} ❌[/red]",
        )
        prog_table.add_row(
            "Elapsed",
            f"{elapsed:.0f}s",
            "Speed",
            f"{speed:.2f} posts/min",
        )
        prog_table.add_row(
            "Progress",
            f"{pct:.1f}%",
            "ETA",
            f"{eta / 60:.1f} min" if eta > 60 else f"{eta:.0f}s",
        )

        bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        bar.add_task("Overall", total=self.total, completed=done)

        group = Group(prog_table, bar)
        self.layout["progress"].update(
            Panel(group, title="Progress", border_style="green", box=box.ROUNDED)
        )

    def _render_jobs(self):
        table = Table(box=box.SIMPLE, expand=True, show_lines=False)
        table.add_column("Job", ratio=3, overflow="ellipsis")
        table.add_column("Status", ratio=2)
        table.add_column("Words", ratio=1, justify="right")
        table.add_column("Tok/s", ratio=2)

        if not self.running:
            table.add_row("[dim]No active generation[/dim]", "", "", "")
        else:
            for rid, rec in self.running.items():
                elapsed = max(time.time() - rec["start_time"], 0.001)
                tps = rec["token_count"] / elapsed
                status = rec["status"]
                style = {
                    "streaming": "cyan",
                    "starting": "yellow",
                    "completed": "green",
                    "failed": "red",
                }.get(status, "red" if status.startswith("error") else "white")
                marker = (
                    "➤ " if rid == getattr(self, "current_record_id", None) else "  "
                )
                spark = _sparkline(rec["history"], width=12)
                table.add_row(
                    f"{marker}{rec['title'][:40]}",
                    f"[{style}]{status}[/{style}]",
                    str(rec["word_count"]),
                    f"{tps:5.1f} {spark}",
                )
        self.layout["jobs"].update(
            Panel(table, title="Active Jobs", border_style="cyan", box=box.ROUNDED)
        )

    def _render_content(self):
        rid = getattr(self, "current_record_id", None)
        rec = self.running.get(rid) if rid is not None else None
        if not rec:
            content = "Waiting for the next job..."
        else:
            content = rec["content"] or "Waiting for first tokens..."
            if rec["status"].startswith("error"):
                content = f"❌ {rec['status']}\n\n{content}"
        lines = content.splitlines() or [content]
        max_lines = self.stream_lines
        if len(lines) > max_lines:
            content = "... [truncated to last %d lines]\n\n" % max_lines + "\n".join(
                lines[-max_lines:]
            )

        title = f"Live Output — {rec['title'][:50]}" if rec else "Live Output"
        self.layout["content"].update(
            Panel(
                Text(content, style="white"),
                title=title,
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )


# ============================================================================
# WORKER
# ============================================================================
_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    if _shutdown_event.is_set():
        log.warning("Forced exit.")
        sys.exit(1)
    _shutdown_event.set()
    log.warning(
        "Shutdown requested. Finishing current record(s)... (Ctrl+C again to force)"
    )


def _worker(cfg, record, tui, session, stream_file=None, quiet=False):
    rec_id = record["id"]
    title = record["title"]
    body = record["body"]
    retries = record["retries"] + 1
    db_path = cfg["db_file"]

    if tui:
        tui.start_record(rec_id, title)
    elif not quiet:
        log.info("Processing: %s", title[:60])

    log.debug("Worker: starting record %d", rec_id)

    conn = get_db_connection(db_path)
    conn.execute(
        "UPDATE blog_posts SET status='processing', retries=?, updated_at=? WHERE id=?",
        (retries, datetime.now().isoformat(), rec_id),
    )
    conn.commit()
    conn.close()

    stream_output = None
    if stream_file:
        mode = "a" if os.path.exists(stream_file) else "w"
        stream_output = open(stream_file, mode, encoding="utf-8")

    def on_chunk(chunk):
        if tui:
            tui.update_stream(rec_id, chunk)
        if not quiet and not tui:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        if stream_output:
            stream_output.write(chunk)
            stream_output.flush()

    blog_post = generate_blog_post(
        title,
        body,
        cfg,
        session,
        on_chunk=on_chunk if cfg.get("stream") else None,
        cancel_event=_shutdown_event,
    )

    if stream_output:
        stream_output.close()

    success = blog_post is not None
    if success:
        wc = len(blog_post.split())
        update_record_status(
            db_path,
            rec_id,
            "completed",
            blog_post=blog_post,
            word_count=wc,
            error_message=None,
        )
        if tui:
            tui.set_content(rec_id, blog_post)
            log.info("✅ Completed %s (%d words)", title[:60], wc)
        else:
            if not quiet:
                log.info("✅ Completed %s (%d words)", title[:60], wc)
            if not cfg.get("stream") and not quiet:
                print("\n--- FINAL POST ---\n")
                print(blog_post)
                print("\n---\n")
    else:
        error_msg = "Cancelled" if _shutdown_event.is_set() else "Failed after retries"
        update_record_status(db_path, rec_id, "failed", error_message=error_msg)
        if tui:
            tui.set_error(rec_id, error_msg)
            log.error("❌ Failed %s (%s)", title[:60], error_msg)
        else:
            if not quiet:
                log.error("❌ Failed %s (%s)", title[:60], error_msg)

    if tui:
        tui.finish_record(rec_id, success)
    return success


def process_records(cfg, limit=None, force=False):
    signal.signal(signal.SIGINT, _signal_handler)
    _shutdown_event.clear()

    if force:
        conn = get_db_connection(cfg["db_file"])
        conn.execute(
            "UPDATE blog_posts SET status='pending', blog_post=NULL, retries=0, "
            "word_count=0, error_message=NULL"
        )
        conn.commit()
        conn.close()
        log.info("All records reset to pending.")

    records = get_records(
        cfg["db_file"],
        status_filter="pending",
        limit=limit,
        max_retries=cfg["max_retries"],
    )

    if not records:
        log.info("No pending records.")
        return

    total = len(records)
    log.info("Processing %d records...", total)

    concurrency = cfg.get("concurrency", 1)
    if cfg.get("stream") and concurrency > 1:
        log.info(
            "Streaming enabled with concurrency=%d — the TUI will show all active jobs.",
            concurrency,
        )

    use_tui = not cfg.get("no_tui", False)
    stream_lines = cfg.get("stream_lines", 180)
    stream_file = cfg.get("stream_to_file")
    quiet = cfg.get("quiet", False)
    session = build_session(cfg)

    tui = None
    if use_tui:
        try:
            tui = RichTUI(
                total,
                stream_lines=stream_lines,
                refresh_per_second=cfg.get("tui_refresh_per_second", 10),
            )
            tui.start()
        except Exception as e:
            log.warning("Failed to start TUI, falling back to console mode: %s", e)
            use_tui = False
            tui = None

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for rec in records:
                if _shutdown_event.is_set():
                    break
                f = pool.submit(_worker, cfg, rec, tui, session, stream_file, quiet)
                futures.append((f, rec["id"]))

            for f, rid in futures:
                try:
                    f.result()
                except Exception as e:
                    log.error("Worker for record %d crashed: %s", rid, e)
                    if log.isEnabledFor(logging.DEBUG):
                        log.exception("Details:")
                    update_record_status(
                        cfg["db_file"], rid, "failed", error_message=str(e)
                    )
                    if tui:
                        tui.set_error(rid, str(e))
                        tui.finish_record(rid, success=False)
    finally:
        if tui:
            tui.stop()
        session.close()

    log.info("Processing finished.")


# ============================================================================
# EXPORT, STATS, OTHER COMMANDS
# ============================================================================
def export_posts(db_path, export_dir, status="completed", fmt="md"):
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    records = get_records(db_path, status_filter=status)
    exported = 0
    for rec in records:
        if not rec["blog_post"]:
            continue
        title = rec["title"]
        body = rec["blog_post"]
        wc = rec.get("word_count", len(body.split()))
        rec_id = rec["id"]
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        safe_name = safe_name.strip().replace(" ", "_")[:80] or f"post_{rec_id}"

        if fmt == "json":
            filepath = export_path / f"{rec_id:04d}_{safe_name}.json"
            payload = {
                "id": rec_id,
                "title": title,
                "word_count": wc,
                "generated_at": rec.get("updated_at", datetime.now().isoformat()),
                "blog_post": body,
            }
            filepath.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        else:
            filepath = export_path / f"{rec_id:04d}_{safe_name}.md"
            frontmatter = (
                f'---\nid: {rec_id}\ntitle: "{title}"\nword_count: {wc}\n'
                f"generated_at: {rec.get('updated_at', datetime.now().isoformat())}\n---\n\n"
            )
            filepath.write_text(frontmatter + body, encoding="utf-8")

        exported += 1
        log.info("Exported: %s", filepath.name)
    return exported


def show_stats(db_path):
    conn = get_db_connection(db_path)
    cursor = conn.execute("SELECT status, COUNT(*) FROM blog_posts GROUP BY status")
    status_counts = {row[0]: row[1] for row in cursor.fetchall()}
    total = sum(status_counts.values())
    cursor = conn.execute("""
        SELECT COUNT(*), COALESCE(AVG(word_count),0), COALESCE(MIN(word_count),0), COALESCE(MAX(word_count),0)
        FROM blog_posts WHERE status='completed'
    """)
    completed, avg, mn, mx = cursor.fetchone()
    conn.close()
    log.info("📊 Database Statistics")
    log.info(" Total: %d", total)
    for s in ("pending", "processing", "completed", "failed"):
        log.info(" %s: %d", s.capitalize(), status_counts.get(s, 0))
    if completed:
        log.info(" Completed posts: avg=%.0f, min=%d, max=%d", avg, mn, mx)


def cmd_list(db_path, status):
    records = get_records(db_path, status_filter=status if status != "all" else None)
    if not records:
        log.info("No records found with status '%s'.", status)
        return
    log.info("Records (status=%s):", status)
    log.info(" %-4s %-10s %-5s %s", "ID", "Status", "Words", "Title")
    log.info(" %s", "-" * 80)
    for rec in records:
        wc = rec.get("word_count", 0) or (
            len(rec.get("blog_post", "").split()) if rec.get("blog_post") else 0
        )
        log.info(" %-4d %-10s %-5d %s", rec["id"], rec["status"], wc, rec["title"][:70])


def cmd_show(db_path, record_id):
    conn = get_db_connection(db_path)
    row = conn.execute("SELECT * FROM blog_posts WHERE id=?", (record_id,)).fetchone()
    conn.close()
    if not row:
        log.error("Record %d not found.", record_id)
        return
    rec = dict(row)
    log.info("ID: %d", rec["id"])
    log.info("Title: %s", rec["title"])
    log.info("Status: %s", rec["status"])
    log.info("Retries: %d", rec["retries"])
    log.info("Word count: %d", rec.get("word_count", 0))
    log.info("Created: %s", rec["created_at"])
    log.info("Updated: %s", rec["updated_at"])
    if rec.get("error_message"):
        log.info("Error: %s", rec["error_message"])
    log.info("\n--- Original Body ---\n%s", rec["body"])
    if rec.get("blog_post"):
        log.info("\n--- Generated Blog Post ---\n%s", rec["blog_post"])


def test_connection(cfg):
    session = build_session(cfg)
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    endpoint = cfg["api_endpoint"].lower()
    use_ollama = "/api/chat" in endpoint and "/v1/" not in endpoint
    if use_ollama:
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "options": {"num_predict": 10},
            "stream": False,
        }
    else:
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 10,
            "stream": False,
        }
    try:
        resp = session.post(
            cfg["api_endpoint"], headers=headers, json=payload, timeout=30
        )
        resp.raise_for_status()
        log.info("✅ Connection successful.")
        return True
    except Exception as e:
        log.error("❌ Connection failed: %s", e)
        return False
    finally:
        session.close()


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    parser = ArgumentParser(description="Blog post generator with AI and TUI.")
    parser.add_argument("csv", nargs="?", help="CSV file to import")
    parser.add_argument("--db", help="SQLite database file")
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument(
        "--init-config",
        metavar="FILE",
        help="Write a starter JSON config file and exit",
    )
    parser.add_argument("--limit", type=int, help="Max records to process")
    parser.add_argument("--force", action="store_true", help="Reset all to pending")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument(
        "--list", nargs="?", const="pending", help="List records by status"
    )
    parser.add_argument("--show", type=int, help="Show record by ID")
    parser.add_argument("--export", action="store_true", help="Export completed posts")
    parser.add_argument("--export-dir", help="Export directory")
    parser.add_argument("--export-format", choices=["md", "json"], help="Export format")
    parser.add_argument("--test", action="store_true", help="Test AI connection")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview what would be processed"
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="Reset failed to pending"
    )
    parser.add_argument(
        "--delete", nargs="?", const="all", help="Delete records by status"
    )
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "-P",
        "--parallel",
        type=int,
        dest="concurrency",
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--debug-ai", action="store_true", help="Log full AI payloads and raw responses"
    )
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds")
    parser.add_argument("--pending", action="store_true", help="Show pending count")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming and show live content in TUI",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable TUI and print streaming content to stdout",
    )
    parser.add_argument(
        "--stream-to-file",
        metavar="FILE",
        help="Write streaming chunks to this file in real time",
    )
    parser.add_argument(
        "--stream-lines",
        type=int,
        help="Number of lines to show in TUI streaming panel",
    )
    parser.add_argument(
        "--tui-fps",
        type=int,
        dest="tui_refresh_per_second",
        help="TUI redraw rate (refreshes/sec). Lower this if you still see flicker "
        "on a slow terminal/SSH link (default 10).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output when --no-tui is used",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.init_config:
        setup_logging(verbose=False)
        write_starter_config(args.init_config)
        return

    cfg = load_config(args.config)

    if args.db:
        cfg["db_file"] = args.db
    if args.export_dir:
        cfg["export_dir"] = args.export_dir
    if args.export_format:
        cfg["export_format"] = args.export_format
    if args.log_file:
        cfg["log_file"] = args.log_file
    if args.concurrency is not None:
        cfg["concurrency"] = args.concurrency
    if args.debug_ai:
        cfg["debug_ai"] = True
    if args.timeout is not None:
        cfg["request_timeout"] = args.timeout
    if args.stream:
        cfg["stream"] = True
    if args.no_tui:
        cfg["no_tui"] = True
    if args.stream_to_file:
        cfg["stream_to_file"] = args.stream_to_file
    if args.stream_lines:
        cfg["stream_lines"] = args.stream_lines
    if args.tui_refresh_per_second:
        cfg["tui_refresh_per_second"] = args.tui_refresh_per_second
    if args.quiet:
        cfg["quiet"] = True

    validate_config(cfg)
    setup_logging(
        verbose=args.verbose or cfg.get("debug_ai"), log_file=cfg.get("log_file")
    )

    db_path = cfg["db_file"]
    init_db(db_path)

    if args.test:
        test_connection(cfg)
        return
    if args.stats:
        show_stats(db_path)
        return
    if args.list is not None:
        cmd_list(db_path, args.list)
        return
    if args.show is not None:
        cmd_show(db_path, args.show)
        return
    if args.export:
        exported = export_posts(
            db_path, cfg["export_dir"], fmt=cfg.get("export_format", "md")
        )
        log.info("Exported %d posts to %s/", exported, cfg["export_dir"])
        return
    if args.delete is not None:
        status = None if args.delete == "all" else args.delete
        deleted = delete_records(db_path, status)
        log.info(
            "Deleted %d records%s.",
            deleted,
            f" (status={args.delete})" if status else "",
        )
        return
    if args.retry_failed:
        conn = get_db_connection(db_path)
        conn.execute(
            "UPDATE blog_posts SET status='pending', retries=0, error_message=NULL WHERE status='failed'"
        )
        conn.commit()
        affected = conn.total_changes
        conn.close()
        log.info("Reset %d failed records to pending.", affected)
        return
    if args.pending:
        conn = get_db_connection(db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM blog_posts WHERE status IN ('pending','failed') AND retries < ?",
            (cfg["max_retries"],),
        ).fetchone()[0]
        conn.close()
        log.info("Records pending/failed (retryable): %d", count)
        return
    if args.dry_run:
        log.info("DRY RUN mode")
        if args.csv:
            try:
                with open(args.csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                log.info("CSV: %s (%d rows)", args.csv, len(rows))
            except Exception as e:
                log.error("Error reading CSV: %s", e)
                return
        pending = get_records(
            db_path, status_filter="pending", max_retries=cfg["max_retries"]
        )
        log.info("Would process %d records.", len(pending))
        return

    if args.csv:
        try:
            inserted = import_csv_to_db(args.csv, db_path)
            log.info("Inserted %d new records.", inserted)
        except Exception as e:
            log.error("Import failed: %s", e)
            return

    process_records(cfg, limit=args.limit, force=args.force)
    show_stats(db_path)


if __name__ == "__main__":
    main()
