#!/usr/bin/env python3
"""
build_tipranks_cache_same_logic.py

Standalone TipRanks cache builder.

This intentionally uses the same TipRanks request/parsing logic as your working
screener_better_zen.py, but only stores TipRanks Smart Score values into a CSV.

Install:
  pip install curl_cffi beautifulsoup4 requests

Examples:
  python build_tipranks_cache_same_logic.py --file tickers.txt
  python build_tipranks_cache_same_logic.py --file tickers.txt --workers 8
  python build_tipranks_cache_same_logic.py --file tickers_check.txt --limit 10 --workers 8

Outputs:
  tipranks_cache.csv
      ticker,tip_ranks

  tipranks_cache_log.csv
      ticker,status,tip_ranks,tipranks_url,error

Notes:
  - This does not query Zacks or WallStreetZen.
  - This does not rotate/change IPs or bypass blocking.
  - It uses the same browser-like headers, curl_cffi Chrome impersonation,
    request_get(), html_to_text(), and get_tipranks_score() logic style from
    the working screener.
  - Existing successful rows in tipranks_cache.csv are loaded and skipped by
    default, so you can stop/restart safely.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

try:
    # Same dependency pattern as the working screener.
    from curl_cffi import requests as http
    HAS_CURL_CFFI = True
except Exception:
    import requests as http  # type: ignore
    HAS_CURL_CFFI = False


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}


DEFAULT_DOMAIN_DELAYS = {
    # Same default as your current screener for TipRanks.
    "www.tipranks.com": 1.25,
}


class DomainRateLimiter:
    """
    Thread-safe minimum delay between requests to the same domain.

    Same idea as the working screener: workers can run in parallel, but requests
    to the same domain are spaced out and shared cooldown is respected.
    """

    def __init__(self, default_delay: float, domain_delays: Optional[dict[str, float]] = None):
        self.default_delay = max(0.0, default_delay)
        self.domain_delays = dict(domain_delays or {})
        self.last_request_at: dict[str, float] = {}
        self.cooldown_until: dict[str, float] = {}
        self.lock = threading.Lock()

    def delay_for(self, host: str) -> float:
        return max(0.0, self.domain_delays.get(host, self.default_delay))

    def wait(self, url: str) -> None:
        host = urlparse(url).netloc.lower()
        delay = self.delay_for(host)

        with self.lock:
            now = time.monotonic()
            normal_slot = self.last_request_at.get(host, 0.0) + delay
            cooldown_slot = self.cooldown_until.get(host, 0.0)
            earliest = max(normal_slot, cooldown_slot)
            sleep_for = max(0.0, earliest - now)
            self.last_request_at[host] = max(now, earliest)

        if sleep_for > 0:
            time.sleep(sleep_for)

    def set_cooldown(self, url: str, seconds: float) -> None:
        if seconds <= 0:
            return

        host = urlparse(url).netloc.lower()
        until = time.monotonic() + seconds

        with self.lock:
            self.cooldown_until[host] = max(self.cooldown_until.get(host, 0.0), until)


REQUEST_LIMITER = DomainRateLimiter(default_delay=0.75, domain_delays=DEFAULT_DOMAIN_DELAYS)
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 2.0


class NonRetryableHTTPError(RuntimeError):
    """HTTP status that should not be retried, such as 403/404."""


def parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None

    value = value.strip()
    if value.isdigit():
        return float(value)

    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            return None
        return max(0.0, retry_at.timestamp() - time.time())
    except Exception:
        return None


def parse_domain_delay_overrides(items: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --domain-delay value: {item!r}. Use domain=seconds.")
        host, raw_delay = item.split("=", 1)
        host = host.strip().lower()
        if not host:
            raise ValueError(f"Bad --domain-delay value: {item!r}. Domain is empty.")
        try:
            delay = float(raw_delay)
        except ValueError as exc:
            raise ValueError(f"Bad delay for {host}: {raw_delay!r}") from exc
        if delay < 0:
            raise ValueError(f"Delay for {host} cannot be negative.")
        overrides[host] = delay

    return overrides


def configure_request_policy(
    request_delay: float,
    domain_delay_items: list[str],
    retries: int,
    backoff: float,
) -> None:
    global REQUEST_LIMITER, REQUEST_RETRIES, REQUEST_BACKOFF

    if request_delay < 0:
        raise ValueError("--request-delay cannot be negative.")
    if retries < 0:
        raise ValueError("--retries cannot be negative.")
    if backoff < 0:
        raise ValueError("--backoff cannot be negative.")

    domain_delays = dict(DEFAULT_DOMAIN_DELAYS)
    domain_delays.update(parse_domain_delay_overrides(domain_delay_items))

    REQUEST_LIMITER = DomainRateLimiter(default_delay=request_delay, domain_delays=domain_delays)
    REQUEST_RETRIES = retries
    REQUEST_BACKOFF = backoff


def request_get(url: str):
    """Same request style as the working screener."""
    kwargs = {"headers": HEADERS, "timeout": 25}
    if HAS_CURL_CFFI:
        kwargs["impersonate"] = "chrome120"

    last_exc: Optional[BaseException] = None
    attempts = REQUEST_RETRIES + 1

    for attempt in range(attempts):
        REQUEST_LIMITER.wait(url)

        try:
            response = http.get(url, **kwargs)

            # Retry transient/rate-limit statuses, but do not hammer the site.
            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt < attempts - 1:
                    retry_after = parse_retry_after_seconds(response.headers.get("Retry-After"))
                    sleep_for = retry_after if retry_after is not None else REQUEST_BACKOFF * (2 ** attempt)
                    sleep_for = min(max(sleep_for, 0.0), 120.0)
                    REQUEST_LIMITER.set_cooldown(url, sleep_for)
                    print(
                        f"Rate limited/transient HTTP {response.status_code} for {url}; "
                        f"cooling down this domain for {sleep_for:.1f}s before retry "
                        f"{attempt + 1}/{REQUEST_RETRIES}",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_for)
                    continue

            # Same policy as the current screener: client errors are not retried.
            if 400 <= response.status_code < 500:
                raise NonRetryableHTTPError(f"HTTP {response.status_code} for {url}")

            response.raise_for_status()
            return response

        except NonRetryableHTTPError:
            raise

        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                break

            sleep_for = min(REQUEST_BACKOFF * (2 ** attempt), 120.0)
            print(
                f"Request failed for {url}; waiting {sleep_for:.1f}s before retry "
                f"{attempt + 1}/{REQUEST_RETRIES}: {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_for)

    assert last_exc is not None
    raise last_exc


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def get_tipranks_score(ticker: str) -> tuple[int, str]:
    """Same TipRanks parser logic as the working screener."""
    ticker_lower = ticker.lower()
    url = f"https://www.tipranks.com/stocks/{ticker_lower}"
    html = request_get(url).text
    text = html_to_text(html)

    idx = text.lower().find("stock smart score")
    if idx != -1:
        chunk = text[idx : idx + 800]
        match = re.search(
            r"Stock Smart Score\s*(10|[1-9])\b",
            chunk,
            flags=re.I,
        )
        if match:
            return int(match.group(1)), url

    # Fallbacks for embedded JSON/state data.
    json_patterns = [
        r'"smartScore"\s*:\s*"?(10|[1-9])"?',
        r'"smart_score"\s*:\s*"?(10|[1-9])"?',
        r'"score"\s*:\s*"?(10|[1-9])"?\s*,\s*"scoreText"',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, html, flags=re.I)
        if match:
            return int(match.group(1)), url

    raise RuntimeError(f"Could not find TipRanks Smart Score for {ticker}.")


def split_ticker_text(text: str) -> list[str]:
    return [part.strip().upper().lstrip("$") for part in re.split(r"[,\s]+", text) if part.strip()]


def read_tickers_from_file(file_path: str) -> list[str]:
    path = Path(file_path.strip().strip('"').strip("'"))
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    tickers: list[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.extend(split_ticker_text(line))

    seen = set()
    unique: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)
    return unique


def read_existing_cache(cache_path: Path) -> dict[str, int]:
    existing: dict[str, int] = {}
    if not cache_path.exists():
        return existing

    with cache_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            raw_score = (row.get("tip_ranks") or row.get("tipranks_score") or "").strip()
            if not ticker or not raw_score:
                continue
            try:
                score = int(float(raw_score))
            except ValueError:
                continue
            if 1 <= score <= 10:
                existing[ticker] = score
    return existing


def ensure_cache_header(cache_path: Path) -> None:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return
    with cache_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "tip_ranks"])


def ensure_log_header(log_path: Path) -> None:
    if log_path.exists() and log_path.stat().st_size > 0:
        return
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "status", "tip_ranks", "tipranks_url", "error"])


@dataclass
class CacheResult:
    ticker: str
    ok: bool
    score: Optional[int]
    url: Optional[str]
    error: Optional[str]


def fetch_one(ticker: str) -> CacheResult:
    try:
        score, url = get_tipranks_score(ticker)
        return CacheResult(ticker=ticker, ok=True, score=score, url=url, error=None)
    except Exception as exc:
        return CacheResult(ticker=ticker, ok=False, score=None, url=None, error=str(exc))


def build_cache(
    tickers: list[str],
    output_path: Path,
    log_path: Path,
    workers: int,
    progress_every: int,
    no_skip_existing: bool,
) -> tuple[int, int, int]:
    existing = {} if no_skip_existing else read_existing_cache(output_path)
    ensure_cache_header(output_path)
    ensure_log_header(log_path)

    todo = [ticker for ticker in tickers if ticker not in existing]
    skipped = len(tickers) - len(todo)

    print(f"Loaded {len(tickers)} unique tickers.", file=sys.stderr)
    print(f"Existing successful TipRanks cache rows: {len(existing)}", file=sys.stderr)
    print(f"Tickers still to fetch: {len(todo)}", file=sys.stderr)
    print(f"Writing success cache to: {output_path}", file=sys.stderr)
    print(f"Writing detailed log to: {log_path}", file=sys.stderr)

    write_lock = threading.Lock()
    success = 0
    failed = 0
    total = len(todo)

    def write_result(result: CacheResult) -> None:
        with write_lock:
            if result.ok:
                with output_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([result.ticker, result.score])
                with log_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([result.ticker, "ok", result.score, result.url, ""])
            else:
                with log_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([result.ticker, "error", "", "", result.error or ""])

    if workers <= 1:
        for index, ticker in enumerate(todo, start=1):
            result = fetch_one(ticker)
            write_result(result)
            if result.ok:
                success += 1
                print(f"OK {index}/{total}: {ticker} TipRanks={result.score}", file=sys.stderr)
            else:
                failed += 1
                print(f"ERROR {index}/{total}: {ticker}: {result.error}", file=sys.stderr)

            if progress_every > 0 and index % progress_every == 0:
                print(f"Progress: {index}/{total}; success={success}; failed={failed}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {executor.submit(fetch_one, ticker): ticker for ticker in todo}
            for completed, future in enumerate(as_completed(future_to_ticker), start=1):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = CacheResult(ticker=ticker, ok=False, score=None, url=None, error=str(exc))

                write_result(result)
                if result.ok:
                    success += 1
                    print(f"OK {completed}/{total}: {ticker} TipRanks={result.score}", file=sys.stderr)
                else:
                    failed += 1
                    print(f"ERROR {completed}/{total}: {ticker}: {result.error}", file=sys.stderr)

                if progress_every > 0 and completed % progress_every == 0:
                    print(f"Progress: {completed}/{total}; success={success}; failed={failed}", file=sys.stderr)

    return success, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a TipRanks cache using the same TipRanks logic as the working screener."
    )
    parser.add_argument(
        "items",
        nargs="*",
        help="Optional ticker symbols. Usually use --file tickers.txt instead.",
    )
    parser.add_argument(
        "--file",
        dest="file_path",
        help="Path to a file containing ticker symbols, one per line or separated by spaces/commas.",
    )
    parser.add_argument(
        "--output",
        default="tipranks_cache.csv",
        help="Output success cache CSV. Default: tipranks_cache.csv",
    )
    parser.add_argument(
        "--log",
        default="tipranks_cache_log.csv",
        help="Detailed log CSV. Default: tipranks_cache_log.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers. Default: 8, same style as your working screener runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N unique tickers. Useful for testing.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.75,
        help="Minimum seconds between requests to the same domain unless overridden. Default: 0.75.",
    )
    parser.add_argument(
        "--domain-delay",
        action="append",
        default=[],
        metavar="DOMAIN=SECONDS",
        help="Override delay for one domain. Example: --domain-delay www.tipranks.com=1.25",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for HTTP 429/5xx or request failures. Default: 3.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Base exponential backoff in seconds after failed/rate-limited requests. Default: 2.0.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N completed tickers. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip tickers already present in the output cache.",
    )
    args = parser.parse_args()

    if args.workers <= 0:
        print("ERROR: --workers must be a positive integer.", file=sys.stderr)
        return 1
    if args.workers > 32:
        print("ERROR: --workers above 32 is too aggressive.", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit <= 0:
        print("ERROR: --limit must be a positive integer.", file=sys.stderr)
        return 1
    if args.progress_every < 0:
        print("ERROR: --progress-every cannot be negative.", file=sys.stderr)
        return 1

    try:
        configure_request_policy(
            request_delay=args.request_delay,
            domain_delay_items=args.domain_delay,
            retries=args.retries,
            backoff=args.backoff,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        if args.file_path:
            tickers = read_tickers_from_file(args.file_path)
        else:
            tickers = split_ticker_text(" ".join(args.items))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        tickers = tickers[: args.limit]

    if not tickers:
        print("ERROR: No tickers were entered.", file=sys.stderr)
        return 1

    output_path = Path(args.output).resolve()
    log_path = Path(args.log).resolve()

    success, skipped, failed = build_cache(
        tickers=tickers,
        output_path=output_path,
        log_path=log_path,
        workers=args.workers,
        progress_every=args.progress_every,
        no_skip_existing=args.no_skip_existing,
    )

    print(
        f"Done. success={success}, skipped_existing={skipped}, failed={failed}. "
        f"Cache file: {output_path}",
        file=sys.stderr,
    )

    # Return 0 if at least one success was written or everything was already cached.
    return 0 if success > 0 or skipped > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
