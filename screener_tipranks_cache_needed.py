#!/usr/bin/env python3
"""
screener_from_tipranks_cache_threshold.py

Purpose:
  Use a local TipRanks cache instead of querying TipRanks live.
  Then process stocks one-by-one:
    1) Read TipRanks Smart Score from tipranks_cache.csv.
    2) Query Zacks Rank live.
    3) Compute the Zacks+TipRanks preliminary score.
    4) Query WallStreetZen only if preliminary score >= --zen-threshold.
    5) For stocks where Zacks, TipRanks, and Zen are all available, compute the full final score.

Your full formula:

perfect_count = I(zacks_rank=1) + I(zen_rank=1) + I(tip_ranks=10)

zacks_points = 2 * (6 - zacks_rank)
zen_points   = 2 * (6 - zen_rank)

score = -6
        + zacks_points
        + zen_points
        + tip_ranks
        + I(zacks_rank=1)
        + I(zen_rank=1)
        + I(tip_ranks=10)
        + I(tip_ranks>=9)
        + 2*I(perfect_count >= 2)

Zen filtering rule in this script:

zacks_tipranks_score = -6
                       + zacks_points
                       + tip_ranks
                       + I(zacks_rank=1)
                       + I(tip_ranks=10)
                       + I(tip_ranks>=9)
                       + 2*I(partial_perfect_count >= 2)

where partial_perfect_count = I(zacks_rank=1) + I(tip_ranks=10).

If zacks_tipranks_score >= --zen-threshold, query Zen.
Default threshold: 6.

Install:
  pip install curl_cffi beautifulsoup4 requests

Example:
  python screener_from_tipranks_cache_threshold.py --top 100 --file tickers.txt --tipranks-cache tipranks_cache.csv --zen-threshold 6 --zen-exchanges nasdaq,nyse --domain-delay www.wallstreetzen.com=80

Outputs:
  scores.csv      -> all processed tickers
  top_scores.csv  -> only fully scored tickers, sorted by final score

Notes:
  - This script does NOT query TipRanks live.
  - This script does NOT rotate/change IPs.
  - It uses the same Zacks/Zen request/parsing style as your working screener.
  - It processes stock-by-stock intentionally, so Zen is not hit in parallel.
"""

from __future__ import annotations

import argparse
import csv
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
    "quote-feed.zacks.com": 0.75,
    "www.zacks.com": 1.00,
    "www.wallstreetzen.com": 3.00,
}


class DomainRateLimiter:
    """Thread-safe minimum delay between requests to the same domain."""

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
    """HTTP status that should not be retried, such as 404 Not Found."""


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
    """GET with per-domain throttling and backoff for rate-limit responses."""
    kwargs = {"headers": HEADERS, "timeout": 25}
    if HAS_CURL_CFFI:
        kwargs["impersonate"] = "chrome120"

    last_exc: Optional[BaseException] = None
    attempts = REQUEST_RETRIES + 1

    for attempt in range(attempts):
        REQUEST_LIMITER.wait(url)

        try:
            response = http.get(url, **kwargs)

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


def clean_json_maybe_jsonp(text: str) -> dict:
    text = text.strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    return json.loads(text)


ZEN_LETTER_TO_RANK = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "F": 5,
}

ZEN_TEXT_TO_RANK = {
    "strong buy": 1,
    "buy": 2,
    "hold": 3,
    "strong sell": 5,
    "sell": 4,
}

VALID_ZEN_EXCHANGES = ["nasdaq", "nyse", "amex", "otc"]


def parse_zen_exchange_order(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None

    exchanges = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not exchanges:
        raise ValueError("--zen-exchanges cannot be empty.")

    invalid = [exchange for exchange in exchanges if exchange not in VALID_ZEN_EXCHANGES]
    if invalid:
        raise ValueError(
            "Bad --zen-exchanges value(s): "
            + ", ".join(invalid)
            + ". Allowed: "
            + ", ".join(VALID_ZEN_EXCHANGES)
        )

    unique: list[str] = []
    seen = set()
    for exchange in exchanges:
        if exchange not in seen:
            unique.append(exchange)
            seen.add(exchange)
    return unique


def get_zacks_rank(ticker: str) -> tuple[int, Optional[str]]:
    ticker = ticker.upper()

    endpoint = f"https://quote-feed.zacks.com/index?t={ticker}"
    try:
        data = clean_json_maybe_jsonp(request_get(endpoint).text)
        row = data.get(ticker)
        if row:
            rank_raw = row.get("zacks_rank") or row.get("zacksRank")
            text_raw = row.get("zacks_rank_text") or row.get("zacksRankText")
            rank = int(rank_raw)
            if rank in {1, 2, 3, 4, 5}:
                return rank, str(text_raw) if text_raw is not None else None
    except Exception:
        pass

    url = f"https://www.zacks.com/stock/quote/{ticker}"
    text = html_to_text(request_get(url).text)
    chunk_start = text.lower().find("zacks rank")
    chunk = text[chunk_start : chunk_start + 1500] if chunk_start != -1 else text

    match = re.search(
        r"\b([1-5])\s*[-–]\s*(Strong Buy|Buy|Hold|Sell|Strong Sell)\s+of\s+5\b",
        chunk,
        flags=re.I,
    )
    if not match:
        match = re.search(r"Zacks Rank\D{0,80}([1-5])\b", chunk, flags=re.I | re.S)

    if not match:
        raise RuntimeError(f"Could not find Zacks Rank for {ticker}.")

    rank = int(match.group(1))
    text_rank = match.group(2).title() if match.lastindex and match.lastindex >= 2 else None
    return rank, text_rank


def normalize_wszen_line(line: str) -> str:
    return re.sub(r"^[#\s]+", "", line.strip()).strip().lower()


def extract_main_zen_rating_from_text(text: str) -> tuple[int, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if normalize_wszen_line(line) != "zen rating":
            continue

        block = lines[i + 1 : i + 40]
        stop_words = {
            "zen rating component grades",
            "industry rating",
            "name / ticker price zen rating",
            "overview",
            "due diligence score",
        }

        trimmed_block: list[str] = []
        for block_line in block:
            normalized = normalize_wszen_line(block_line)
            if normalized in stop_words:
                break
            trimmed_block.append(block_line)

        block_text = "\n".join(trimmed_block)

        match = re.search(
            r"(?<![A-Z])([ABCDF])\s+(Strong Buy|Buy|Hold|Strong Sell|Sell)(?![a-z])",
            block_text,
            flags=re.I,
        )
        if match:
            letter = match.group(1).upper()
            label = match.group(2).title()
            return ZEN_LETTER_TO_RANK[letter], f"{letter} {label}"

        for j in range(len(trimmed_block) - 1):
            letter = trimmed_block[j].strip().upper()
            label = trimmed_block[j + 1].strip().lower()
            if letter in ZEN_LETTER_TO_RANK and label in ZEN_TEXT_TO_RANK:
                return ZEN_LETTER_TO_RANK[letter], f"{letter} {trimmed_block[j + 1].title()}"

    raise RuntimeError("main Zen Rating block was not found")


def get_zen_rank(
    ticker: str,
    preferred_exchange: Optional[str] = None,
    exchange_order: Optional[list[str]] = None,
) -> tuple[int, str, str]:
    ticker_lower = ticker.lower()

    if preferred_exchange:
        exchanges = [preferred_exchange.lower()]
    elif exchange_order:
        exchanges = exchange_order
    else:
        exchanges = VALID_ZEN_EXCHANGES

    errors: list[str] = []
    for exchange in exchanges:
        url = f"https://www.wallstreetzen.com/stocks/us/{exchange}/{ticker_lower}"
        try:
            response = request_get(url)
            text = html_to_text(response.text)
            rank, rating = extract_main_zen_rating_from_text(text)
            return rank, rating, url
        except Exception as exc:
            errors.append(f"{exchange}: {exc}")
            continue

    detail = " Errors: " + " | ".join(errors) if errors else ""
    raise RuntimeError(
        f"Could not find WallStreetZen Zen Rating for {ticker}. "
        f"Try passing --exchange nasdaq/nyse/amex/otc.{detail}"
    )


def clean_ticker(ticker: str) -> str:
    return ticker.strip().upper().lstrip("$")


def split_ticker_text(text: str) -> list[str]:
    return [clean_ticker(part) for part in re.split(r"[,\s]+", text) if part.strip()]


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
        if ticker and ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)
    return unique


def parse_int_score(value: str) -> Optional[int]:
    value = str(value).strip()
    if not value or value.upper() in {"NA", "N/A", "NONE", "NULL", "-"}:
        return None
    match = re.search(r"\b(10|[1-9])\b", value)
    if not match:
        return None
    score = int(match.group(1))
    if 1 <= score <= 10:
        return score
    return None


def load_tipranks_cache(cache_path: str) -> dict[str, int]:
    path = Path(cache_path.strip().strip('"').strip("'"))
    if not path.exists():
        raise FileNotFoundError(f"TipRanks cache not found: {path}")

    cache: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else True

        if has_header:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return cache
            normalized_fields = {name.strip().lower(): name for name in reader.fieldnames if name}

            ticker_field = (
                normalized_fields.get("ticker")
                or normalized_fields.get("symbol")
                or normalized_fields.get("stock")
            )
            score_field = (
                normalized_fields.get("tip_ranks")
                or normalized_fields.get("tipranks")
                or normalized_fields.get("tipranks_score")
                or normalized_fields.get("smart_score")
                or normalized_fields.get("smartscore")
                or normalized_fields.get("rank")
                or normalized_fields.get("score")
            )

            if not ticker_field or not score_field:
                raise ValueError(
                    "TipRanks cache must have columns like ticker,tip_ranks "
                    "or ticker,rank. Found columns: " + ", ".join(reader.fieldnames)
                )

            for row in reader:
                ticker = clean_ticker(row.get(ticker_field, ""))
                score = parse_int_score(row.get(score_field, ""))
                if ticker and score is not None:
                    cache[ticker] = score
        else:
            reader2 = csv.reader(f)
            for row in reader2:
                if len(row) < 2:
                    continue
                ticker = clean_ticker(row[0])
                score = parse_int_score(row[1])
                if ticker and score is not None:
                    cache[ticker] = score

    return cache


def parse_mode(items: list[str], file_arg: Optional[str]) -> list[str]:
    if file_arg:
        return read_tickers_from_file(file_arg)
    if items:
        return split_ticker_text(" ".join(items))

    print("Enter tickers separated by spaces, OR rerun with --file tickers.txt", file=sys.stderr)
    raw = input("Input: ").strip()
    return split_ticker_text(raw) if raw else []


def compute_score(zacks_rank: int, zen_rank: int, tip_ranks: int) -> int:
    perfect_count = int(zacks_rank == 1) + int(zen_rank == 1) + int(tip_ranks == 10)
    zacks_points = 2 * (6 - zacks_rank)
    zen_points = 2 * (6 - zen_rank)
    return (
        -6
        + zacks_points
        + zen_points
        + tip_ranks
        + int(zacks_rank == 1)
        + int(zen_rank == 1)
        + int(tip_ranks == 10)
        + int(tip_ranks >= 9)
        + 2 * int(perfect_count >= 2)
    )


def zacks_tipranks_score(zacks_rank: int, tip_ranks: int) -> int:
    """
    Preliminary score without Zen, used only to decide whether Zen is worth querying.
    This is exactly the new formula using only Zacks and TipRanks:
      -6 + zacks_points + tip_ranks + Zacks/TipRanks bonuses,
    where zacks_points = 2 * (6 - zacks_rank).
    """
    partial_perfect_count = int(zacks_rank == 1) + int(tip_ranks == 10)
    zacks_points = 2 * (6 - zacks_rank)
    return (
        -6
        + zacks_points
        + tip_ranks
        + int(zacks_rank == 1)
        + int(tip_ranks == 10)
        + int(tip_ranks >= 9)
        + 2 * int(partial_perfect_count >= 2)
    )


def max_possible_score_with_best_zen(zacks_rank: int, tip_ranks: int) -> int:
    """Best possible final score if Zen rank is 1."""
    best_zen_rank = 1
    return compute_score(zacks_rank, best_zen_rank, tip_ranks)


@dataclass
class StockResult:
    ticker: str
    zacks_rank: Optional[int]
    zacks_text: Optional[str]
    tip_ranks: Optional[int]
    zen_rank: Optional[int]
    zen_rating: Optional[str]
    zen_url: Optional[str]
    zacks_tipranks_score: Optional[int]
    max_possible_score: Optional[int]
    final_score: Optional[int]
    queried_zen: bool
    decision: str
    missing_sources: list[str]
    errors: list[str]


def process_ticker(
    ticker: str,
    tipranks_cache: dict[str, int],
    zen_threshold: float,
    preferred_exchange: Optional[str],
    zen_exchanges: Optional[list[str]],
    skip_zen: bool,
) -> StockResult:
    ticker = clean_ticker(ticker)
    missing_sources: list[str] = []
    errors: list[str] = []

    tip_ranks = tipranks_cache.get(ticker)
    if tip_ranks is None:
        missing_sources.append("tipranks_cache")

    try:
        zacks_rank, zacks_text = get_zacks_rank(ticker)
    except Exception as exc:
        zacks_rank, zacks_text = None, None
        missing_sources.append("zacks")
        errors.append(f"Zacks: {exc}")

    preliminary: Optional[int] = None
    max_possible: Optional[int] = None
    queried_zen = False
    zen_rank: Optional[int] = None
    zen_rating: Optional[str] = None
    zen_url: Optional[str] = None
    final_score: Optional[int] = None

    if zacks_rank is None or tip_ranks is None:
        decision = "skip_zen_missing_zacks_or_tipranks"
        return StockResult(
            ticker=ticker,
            zacks_rank=zacks_rank,
            zacks_text=zacks_text,
            tip_ranks=tip_ranks,
            zen_rank=None,
            zen_rating=None,
            zen_url=None,
            zacks_tipranks_score=None,
            max_possible_score=None,
            final_score=None,
            queried_zen=False,
            decision=decision,
            missing_sources=missing_sources,
            errors=errors,
        )

    preliminary = zacks_tipranks_score(zacks_rank, tip_ranks)
    max_possible = max_possible_score_with_best_zen(zacks_rank, tip_ranks)

    if skip_zen:
        decision = "skip_zen_requested"
    elif preliminary < zen_threshold:
        decision = f"skip_zen_preliminary_below_{zen_threshold:g}"
    else:
        queried_zen = True
        try:
            zen_rank, zen_rating, zen_url = get_zen_rank(ticker, preferred_exchange, zen_exchanges)
            final_score = compute_score(zacks_rank, zen_rank, tip_ranks)
            decision = "full_score"
        except Exception as exc:
            missing_sources.append("zen")
            errors.append(f"Zen: {exc}")
            decision = "zen_query_failed"

    return StockResult(
        ticker=ticker,
        zacks_rank=zacks_rank,
        zacks_text=zacks_text,
        tip_ranks=tip_ranks,
        zen_rank=zen_rank,
        zen_rating=zen_rating,
        zen_url=zen_url,
        zacks_tipranks_score=preliminary,
        max_possible_score=max_possible,
        final_score=final_score,
        queried_zen=queried_zen,
        decision=decision,
        missing_sources=missing_sources,
        errors=errors,
    )


def format_optional(value: Optional[int | float | str]) -> str:
    return "NA" if value is None else str(value)


def result_line(result: StockResult) -> str:
    return (
        f"{result.ticker}, {format_optional(result.final_score)} || "
        f"zacks={format_optional(result.zacks_rank)}, "
        f"zen={format_optional(result.zen_rank)}, "
        f"tipranks={format_optional(result.tip_ranks)}, "
        f"zt_pre={format_optional(result.zacks_tipranks_score)}, "
        f"decision={result.decision}"
    )


def write_results_csv(results: list[StockResult], output_path: str) -> Path:
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticker",
            "total_score",
            "zacks_rank",
            "zen_rank",
            "tip_ranks",
            "zacks_tipranks_score",
            "max_possible_score",
            "queried_zen",
            "decision",
            "missing_sources",
            "zacks_text",
            "zen_rating",
            "zen_url",
            "errors",
        ])
        for r in results:
            writer.writerow([
                r.ticker,
                format_optional(r.final_score),
                format_optional(r.zacks_rank),
                format_optional(r.zen_rank),
                format_optional(r.tip_ranks),
                format_optional(r.zacks_tipranks_score),
                format_optional(r.max_possible_score),
                "yes" if r.queried_zen else "no",
                r.decision,
                ";".join(r.missing_sources),
                r.zacks_text or "",
                r.zen_rating or "",
                r.zen_url or "",
                " | ".join(r.errors),
            ])
    return path


def write_top_csv(results: list[StockResult], output_path: str, top_n: Optional[int]) -> Path:
    full = [r for r in results if r.final_score is not None]
    full.sort(key=lambda r: (-int(r.final_score), r.ticker))
    if top_n is not None:
        full = full[:top_n]
    return write_results_csv(full, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score stocks using TipRanks cache, live Zacks, and conditional live Zen."
    )
    parser.add_argument("items", nargs="*", help="Ticker symbols, if --file is not used.")
    parser.add_argument("--file", dest="file_path", help="Path to ticker file, one ticker per line.")
    parser.add_argument(
        "--tipranks-cache",
        default="tipranks_cache.csv",
        help="CSV cache with columns ticker,tip_ranks. Default: tipranks_cache.csv",
    )
    parser.add_argument("--top", type=int, default=100, help="How many fully-scored results to print/write to top CSV. Default: 100.")
    parser.add_argument(
        "--zen-threshold",
        type=float,
        default=6.0,
        help="Query Zen only when Zacks+TipRanks preliminary score is at least this. Default: 6.",
    )
    parser.add_argument("--output", default="scores.csv", help="All-results CSV path. Default: scores.csv")
    parser.add_argument("--top-output", default="top_scores.csv", help="Top full-score CSV path. Default: top_scores.csv")
    parser.add_argument("--limit", type=int, help="Process only the first N tickers. Useful for testing.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Accepted for compatibility. This script intentionally processes stock-by-stock.",
    )
    parser.add_argument(
        "--zen-workers",
        type=int,
        default=1,
        help="Accepted for compatibility. This script intentionally uses one Zen request stream.",
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
        help="Override delay for one domain. Can be repeated. Example: --domain-delay www.wallstreetzen.com=80",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries for HTTP 429/5xx or request failures. Default: 3.")
    parser.add_argument("--backoff", type=float, default=2.0, help="Base exponential backoff seconds. Default: 2.0.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N tickers. Use 0 to disable.")
    parser.add_argument(
        "--exchange",
        choices=VALID_ZEN_EXCHANGES,
        help="Optional WallStreetZen exchange hint. Fastest because it tries only one exchange.",
    )
    parser.add_argument(
        "--zen-exchanges",
        help="Comma-separated Zen exchanges to try when --exchange is not set. Example: nasdaq,nyse.",
    )
    parser.add_argument("--skip-zen", action="store_true", help="Never query Zen. Useful for testing Zacks+TipRanks filtering only.")
    args = parser.parse_args()

    if args.top is not None and args.top <= 0:
        print("ERROR: --top must be positive.", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit <= 0:
        print("ERROR: --limit must be positive.", file=sys.stderr)
        return 1
    if args.progress_every < 0:
        print("ERROR: --progress-every cannot be negative.", file=sys.stderr)
        return 1
    if args.workers != 1:
        print("NOTE: --workers is ignored. This script goes stock-by-stock.", file=sys.stderr)
    if args.zen_workers != 1:
        print("NOTE: --zen-workers is ignored. This script uses one Zen stream.", file=sys.stderr)

    try:
        configure_request_policy(args.request_delay, args.domain_delay, args.retries, args.backoff)
        zen_exchanges = parse_zen_exchange_order(args.zen_exchanges)
        tipranks_cache = load_tipranks_cache(args.tipranks_cache)
        tickers = parse_mode(args.items, args.file_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not tickers:
        print("ERROR: No tickers were entered.", file=sys.stderr)
        return 1
    if args.limit is not None:
        tickers = tickers[: args.limit]

    print(f"Loaded {len(tickers)} unique tickers.", file=sys.stderr)
    print(f"Loaded {len(tipranks_cache)} TipRanks cache rows from {args.tipranks_cache}.", file=sys.stderr)
    print(f"Zen threshold: Zacks+TipRanks preliminary score >= {args.zen_threshold:g}.", file=sys.stderr)

    results: list[StockResult] = []
    total = len(tickers)

    for index, ticker in enumerate(tickers, start=1):
        result = process_ticker(
            ticker=ticker,
            tipranks_cache=tipranks_cache,
            zen_threshold=args.zen_threshold,
            preferred_exchange=args.exchange,
            zen_exchanges=zen_exchanges,
            skip_zen=args.skip_zen,
        )
        results.append(result)

        if result.decision == "full_score":
            print(f"FULL {index}/{total}: {result_line(result)}", file=sys.stderr)
        elif result.queried_zen:
            print(f"ZEN_FAIL {index}/{total}: {result_line(result)}", file=sys.stderr)
        else:
            print(f"SKIP {index}/{total}: {result_line(result)}", file=sys.stderr)

        if args.progress_every > 0 and index % args.progress_every == 0:
            full_count = sum(r.final_score is not None for r in results)
            zen_count = sum(r.queried_zen for r in results)
            print(f"Progress: {index}/{total}; queried_zen={zen_count}; full_scores={full_count}", file=sys.stderr)

    # Sort all results: fully scored first by final score, then potential score, then ticker.
    results.sort(
        key=lambda r: (
            0 if r.final_score is not None else 1,
            -(r.final_score if r.final_score is not None else -999999),
            -(r.max_possible_score if r.max_possible_score is not None else -999999),
            r.ticker,
        )
    )

    full_results = [r for r in results if r.final_score is not None]
    full_results.sort(key=lambda r: (-int(r.final_score), r.ticker))
    top_results = full_results[: args.top] if args.top is not None else full_results

    for r in top_results:
        print(result_line(r))

    all_csv = write_results_csv(results, args.output)
    top_csv = write_top_csv(results, args.top_output, args.top)

    queried_zen = sum(r.queried_zen for r in results)
    skipped_below_threshold = sum(r.decision.startswith("skip_zen_preliminary_below_") for r in results)
    missing_tipranks = sum("tipranks_cache" in r.missing_sources for r in results)
    missing_zacks = sum("zacks" in r.missing_sources for r in results)
    zen_failed = sum(r.decision == "zen_query_failed" for r in results)

    print(f"Saved all results: {all_csv}", file=sys.stderr)
    print(f"Saved top full scores: {top_csv}", file=sys.stderr)
    print(
        "Summary: "
        f"processed={len(results)}, "
        f"queried_zen={queried_zen}, "
        f"full_scores={len(full_results)}, "
        f"skipped_below_threshold={skipped_below_threshold}, "
        f"missing_tipranks_cache={missing_tipranks}, "
        f"missing_zacks={missing_zacks}, "
        f"zen_failed={zen_failed}",
        file=sys.stderr,
    )

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
