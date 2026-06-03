#!/usr/bin/env python3
"""
stock_score.py

Gets these three values for each ticker:
  1) Zacks Rank: 1..5
  2) WallStreetZen / Zen Rating mapped to rank: A=1, B=2, C=3, D=4, F=5
  3) TipRanks Smart Score: 1..10

Then computes your score formula:

perfect_count = I(zacks_rank=1) + I(zen_rank=1) + I(tip_ranks=10)

score = 18 - 2*zacks_rank - 2*zen_rank + tip_ranks
        + I(zacks_rank=1)
        + I(zen_rank=1)
        + I(tip_ranks=10)
        + I(tip_ranks>=9)
        + 2*I(perfect_count >= 2)

Install:
  pip install curl_cffi beautifulsoup4

Run examples:
  python stock_score.py
  python stock_score.py PLTR TSLA NVDA
  python stock_score.py 10 C:\\Users\\yarin\\Desktop\\tickers.txt
  python stock_score.py --top 10 --file C:\\Users\\yarin\\Desktop\\tickers.txt
  python stock_score.py --top 100 --file C:\\Users\\yarin\\Desktop\\tickers.txt --workers 12

Output format:
  ticker, total_score || zacks_rank, zen_rank, tip_ranks

Missing-score fallback:
  - If all 3 values are found, the script uses your normal formula.
  - If exactly 1 value is missing, the script uses the available provider scores
    on a 0..10-ish scale and adds a bonus equal to half of each found provider:
      zacks bonus = (12 - zacks_rank * 2) / 2
      zen bonus   = (12 - zen_rank * 2) / 2
      tip bonus   = tip_ranks / 2
  - If 2 or more values are missing, the missing providers contribute 0.
  - Missing provider values are printed as NA and the ticker is marked "missing a score".
  - Missing-score penalties:
      missing Zacks:    -12
      missing Zen:      -12
      missing TipRanks: -2

CSV output:
  - The same sorted results are also written to scores.csv in the same directory
    as this script.

Notes:
  - No cache is created.
  - For large files, start with --workers 4. Raise carefully only if the sites keep responding.
  - This version intentionally does not rotate/change IPs. It uses per-domain throttling,
    Retry-After handling, and exponential backoff so it behaves better under rate limits.
  - For WallStreetZen specifically, use --exchange when possible, --zen-exchanges to
    reduce exchange guesses, or --zen-top-candidates for large files.
  - This uses public web pages/endpoints. If a site changes its HTML or blocks automated
    requests, the relevant parser may need a small update.
  - Designed mainly for US tickers.
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
    # More reliable against sites that dislike basic Python requests.
    from curl_cffi import requests as http
    HAS_CURL_CFFI = True
except Exception:  # fallback if curl_cffi is not installed
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
    # Conservative defaults: at most about 1 request per domain per second.
    # You can override these from the command line with --domain-delay.
    "quote-feed.zacks.com": 0.75,
    "www.zacks.com": 1.00,
    "www.wallstreetzen.com": 3.00,
    "www.tipranks.com": 1.25,
}


class DomainRateLimiter:
    """
    Thread-safe minimum delay between requests to the same domain.

    Important detail: if one worker receives HTTP 429 with Retry-After, the
    cooldown is shared by all workers. Otherwise, 10 workers can each discover
    the same 429 independently before the first sleeping worker wakes up.
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
            # Reserve the next slot while holding the lock so concurrent workers
            # do not all wake up and hit the same site together.
            self.last_request_at[host] = max(now, earliest)

        if sleep_for > 0:
            time.sleep(sleep_for)

    def set_cooldown(self, url: str, seconds: float) -> None:
        """Pause all future requests to this URL's domain for at least seconds."""
        if seconds <= 0:
            return

        host = urlparse(url).netloc.lower()
        until = time.monotonic() + seconds

        with self.lock:
            self.cooldown_until[host] = max(self.cooldown_until.get(host, 0.0), until)


REQUEST_LIMITER = DomainRateLimiter(default_delay=0.75, domain_delays=DEFAULT_DOMAIN_DELAYS)
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 2.0


def parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
    """Parse Retry-After as either seconds or an HTTP date."""
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
    """Parse values like: www.example.com=2.5"""
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

ZEN_LETTER_TO_RANK = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "F": 5,
}

ZEN_TEXT_TO_RANK = {
    # Order matters if this fallback is ever used.
    "strong buy": 1,
    "buy": 2,
    "hold": 3,
    "strong sell": 5,
    "sell": 4,
}

VALID_ZEN_EXCHANGES = ["nasdaq", "nyse", "amex", "otc"]


def parse_zen_exchange_order(raw: Optional[str]) -> Optional[list[str]]:
    """Parse comma-separated Zen exchanges, for example: nasdaq,nyse."""
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

    # Remove duplicates while preserving order.
    unique: list[str] = []
    seen = set()
    for exchange in exchanges:
        if exchange not in seen:
            unique.append(exchange)
            seen.add(exchange)
    return unique


@dataclass
class RatingResult:
    ticker: str
    zacks_rank: Optional[int]
    zacks_text: Optional[str]
    zen_rank: Optional[int]
    zen_rating: Optional[str]
    zen_url: Optional[str]
    tipranks_score: Optional[int]
    tipranks_url: Optional[str]
    final_score: float
    missing_count: int
    missing_sources: list[str]


class NonRetryableHTTPError(RuntimeError):
    """HTTP status that should not be retried, such as 404 Not Found."""


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

            # Do not retry permanent client errors like 404. Retrying a missing
            # ticker page only wastes time and can increase blocking risk.
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
    # Some endpoints return JSONP-like wrappers. Strip them if present.
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    return json.loads(text)


def get_zacks_rank(ticker: str) -> tuple[int, Optional[str]]:
    ticker = ticker.upper()

    # Endpoint used by the public zacks-api wrapper.
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
        # Fall back to the normal Zacks quote page.
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
    """Normalize lines from either BeautifulSoup text or markdown-like renderers."""
    return re.sub(r"^[#\s]+", "", line.strip()).strip().lower()


def extract_main_zen_rating_from_text(text: str) -> tuple[int, str]:
    """
    Extract only the main WallStreetZen Zen Rating.

    BeautifulSoup returns the heading as "Zen Rating", while some rendered views
    show it as "## Zen Rating". Accept both. Also stop before component grades
    so we do not accidentally read Value/Growth/Sentiment grades.
    """
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

        # Usually appears exactly like: "A Strong Buy", "B Buy", "C Hold", etc.
        match = re.search(
            r"(?<![A-Z])([ABCDF])\s+(Strong Buy|Buy|Hold|Strong Sell|Sell)(?![a-z])",
            block_text,
            flags=re.I,
        )
        if match:
            letter = match.group(1).upper()
            label = match.group(2).title()
            return ZEN_LETTER_TO_RANK[letter], f"{letter} {label}"

        # Fallback: the letter and label may appear on separate lines.
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

def get_tipranks_score(ticker: str) -> tuple[int, str]:
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


def compute_score(zacks_rank: int, zen_rank: int, tip_ranks: int) -> int:
    perfect_count = int(zacks_rank == 1) + int(zen_rank == 1) + int(tip_ranks == 10)
    return (
        18
        - 2 * zacks_rank
        - 2 * zen_rank
        + tip_ranks
        + int(zacks_rank == 1)
        + int(zen_rank == 1)
        + int(tip_ranks == 10)
        + int(tip_ranks >= 9)
        + 2 * int(perfect_count >= 2)
    )


def zacks_fallback_value(zacks_rank: int) -> float:
    # Converts Zacks rank 1..5 to 10, 8, 6, 4, 2.
    return float(12 - zacks_rank * 2)


def zen_fallback_value(zen_rank: int) -> float:
    # Converts Zen rank 1..5 to 10, 8, 6, 4, 2.
    return float(12 - zen_rank * 2)


def tipranks_fallback_value(tip_ranks: int) -> float:
    # TipRanks Smart Score is already on a 1..10 scale.
    return float(tip_ranks)


def missing_score_penalty(
    zacks_rank: Optional[int],
    zen_rank: Optional[int],
    tip_ranks: Optional[int],
) -> float:
    """
    Penalty rules requested by the user:
      missing Zacks    = -12
      missing Zen      = -12
      missing TipRanks = -2
    """
    penalty = 0.0

    if zacks_rank is None:
        penalty -= 12.0
    if zen_rank is None:
        penalty -= 12.0
    if tip_ranks is None:
        penalty -= 2.0

    return penalty


def compute_score_with_fallback(
    zacks_rank: Optional[int],
    zen_rank: Optional[int],
    tip_ranks: Optional[int],
) -> float:
    """
    Scoring rules:
      - 0 missing values: use your original formula.
      - 1 missing value: score the found providers and add your half-score bonus
        from each found provider.
      - 2+ missing values: missing providers contribute 0 and no bonus is added.
      - Then apply missing-score penalties.
    """
    missing_count = sum(value is None for value in (zacks_rank, zen_rank, tip_ranks))

    if missing_count == 0:
        return float(compute_score(zacks_rank, zen_rank, tip_ranks))  # type: ignore[arg-type]

    found_values: list[float] = []
    if zacks_rank is not None:
        found_values.append(zacks_fallback_value(zacks_rank))
    if zen_rank is not None:
        found_values.append(zen_fallback_value(zen_rank))
    if tip_ranks is not None:
        found_values.append(tipranks_fallback_value(tip_ranks))

    score = sum(found_values)

    if missing_count == 1:
        score += sum(value / 2 for value in found_values)

    score += missing_score_penalty(zacks_rank, zen_rank, tip_ranks)
    return score


def score_ticker(
    ticker: str,
    exchange: Optional[str] = None,
    zen_exchanges: Optional[list[str]] = None,
    skip_zen: bool = False,
) -> RatingResult:
    ticker = ticker.upper().strip().lstrip("$")
    if not ticker:
        raise ValueError("empty ticker")

    missing_sources: list[str] = []

    try:
        zacks_rank, zacks_text = get_zacks_rank(ticker)
    except Exception:
        zacks_rank, zacks_text = None, None
        missing_sources.append("zacks")

    if skip_zen:
        zen_rank, zen_rating, zen_url = None, None, None
        missing_sources.append("zen")
    else:
        try:
            zen_rank, zen_rating, zen_url = get_zen_rank(ticker, exchange, zen_exchanges)
        except Exception:
            zen_rank, zen_rating, zen_url = None, None, None
            missing_sources.append("zen")

    try:
        tipranks_score, tipranks_url = get_tipranks_score(ticker)
    except Exception:
        tipranks_score, tipranks_url = None, None
        missing_sources.append("tipranks")

    final_score = compute_score_with_fallback(zacks_rank, zen_rank, tipranks_score)

    return RatingResult(
        ticker=ticker,
        zacks_rank=zacks_rank,
        zacks_text=zacks_text,
        zen_rank=zen_rank,
        zen_rating=zen_rating,
        zen_url=zen_url,
        tipranks_score=tipranks_score,
        tipranks_url=tipranks_url,
        final_score=final_score,
        missing_count=len(missing_sources),
        missing_sources=missing_sources,
    )


def format_score(score: float) -> str:
    if score.is_integer():
        return str(int(score))
    return f"{score:.1f}".rstrip("0").rstrip(".")


def format_rank(value: Optional[int]) -> str:
    return "NA" if value is None else str(value)


def result_line(result: RatingResult) -> str:
    ticker_label = result.ticker
    if result.missing_count > 0:
        ticker_label += " (missing a score)"

    return (
        f"{ticker_label}, {format_score(result.final_score)} || "
        f"{format_rank(result.zacks_rank)}, "
        f"{format_rank(result.zen_rank)}, "
        f"{format_rank(result.tipranks_score)}"
    )


def write_scores_csv(results: list[RatingResult]) -> Path:
    """Write the same sorted results shown in the terminal to scores.csv."""
    output_path = Path(__file__).resolve().parent / "scores.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticker",
            "total_score",
            "zacks_rank",
            "zen_rank",
            "tip_ranks",
            "missing_a_score",
            "missing_sources",
        ])

        for result in results:
            writer.writerow([
                result.ticker,
                format_score(result.final_score),
                format_rank(result.zacks_rank),
                format_rank(result.zen_rank),
                format_rank(result.tipranks_score),
                "yes" if result.missing_count > 0 else "no",
                ";".join(result.missing_sources),
            ])

    return output_path


def split_ticker_text(text: str) -> list[str]:
    # Supports: "PLTR TSLA NVDA" and "PLTR, TSLA, NVDA".
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
            # File is expected to be one ticker per line, but this also tolerates commas/spaces.
            tickers.extend(split_ticker_text(line))

    # Remove duplicates while preserving order.
    seen = set()
    unique: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)
    return unique


def parse_mode(items: list[str], top_arg: Optional[int], file_arg: Optional[str]) -> tuple[list[str], Optional[int]]:
    """
    Returns: (tickers, top_n)

    Supported modes:
      python stock_score.py PLTR TSLA NVDA
      python stock_score.py 10 C:\\path\\tickers.txt
      python stock_score.py top 10 C:\\path\\tickers.txt
      python stock_score.py --top 10 --file C:\\path\\tickers.txt
    """
    if file_arg:
        tickers = read_tickers_from_file(file_arg)
        return tickers, top_arg

    if not items:
        print("Enter tickers separated by spaces, OR enter: number file_path")
        print("Example tickers: PLTR TSLA NVDA")
        print(r"Example file mode: 10 C:\Users\yarin\Desktop\tickers.txt")
        raw = input("Input: ").strip()
        if not raw:
            return [], None
        items = raw.split()

    # File mode: "10 C:\path\tickers.txt"
    if len(items) >= 2 and items[0].isdigit():
        top_n = int(items[0])
        file_path = " ".join(items[1:])
        return read_tickers_from_file(file_path), top_n

    # File mode: "top 10 C:\path\tickers.txt"
    if len(items) >= 3 and items[0].lower() == "top" and items[1].isdigit():
        top_n = int(items[1])
        file_path = " ".join(items[2:])
        return read_tickers_from_file(file_path), top_n

    # Direct ticker mode.
    tickers = split_ticker_text(" ".join(items))
    return tickers, top_arg


def score_many(
    tickers: list[str],
    exchange: Optional[str],
    zen_exchanges: Optional[list[str]],
    workers: int,
    progress_every: int,
    skip_zen: bool = False,
) -> tuple[list[RatingResult], list[str]]:
    """
    Score many tickers concurrently.

    This is the main speed improvement for large input files. The script still does
    not create any cache; it simply looks up several tickers at the same time.
    """
    results: list[RatingResult] = []
    errors: list[str] = []
    total = len(tickers)

    def run_one(ticker: str) -> tuple[Optional[RatingResult], Optional[str]]:
        try:
            return score_ticker(ticker, exchange, zen_exchanges, skip_zen), None
        except Exception as exc:
            return None, f"{ticker}: {exc}"

    if workers <= 1:
        for index, ticker in enumerate(tickers, start=1):
            result, error = run_one(ticker)
            if result is not None:
                results.append(result)
            if error is not None:
                errors.append(error)
            if progress_every > 0 and index % progress_every == 0:
                print(f"Progress: {index}/{total}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {executor.submit(run_one, ticker): ticker for ticker in tickers}

            for completed, future in enumerate(as_completed(future_to_ticker), start=1):
                result, error = future.result()
                if result is not None:
                    results.append(result)
                if error is not None:
                    errors.append(error)

                if progress_every > 0 and completed % progress_every == 0:
                    print(f"Progress: {completed}/{total}", file=sys.stderr)

    results.sort(key=lambda r: (-r.final_score, r.ticker))
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score tickers using Zacks, Zen Rating, and TipRanks."
    )
    parser.add_argument(
        "items",
        nargs="*",
        help=(
            "Ticker symbols, or file mode as: TOP_N FILE_PATH. "
            "Examples: PLTR TSLA NVDA  OR  10 C:\\Users\\yarin\\Desktop\\tickers.txt"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Show only the top X scores. Usually used with --file, but also works with direct tickers.",
    )
    parser.add_argument(
        "--file",
        dest="file_path",
        help="Path to a file containing ticker symbols, one per line.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Number of tickers to query in parallel. Default: 4. "
            "Higher is faster but more likely to trigger rate limits."
        ),
    )
    parser.add_argument(
        "--zen-workers",
        type=int,
        default=1,
        help=(
            "Workers to use in the Zen second pass when --zen-top-candidates is set. "
            "Default: 1. Keep this low because Zen is the site that rate-limits."
        ),
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.75,
        help=(
            "Minimum seconds between requests to the same domain unless overridden. "
            "Default: 0.75. Use a bigger value if you still get blocked."
        ),
    )
    parser.add_argument(
        "--domain-delay",
        action="append",
        default=[],
        metavar="DOMAIN=SECONDS",
        help=(
            "Override delay for one domain. Can be repeated. Example: "
            "--domain-delay www.tipranks.com=3"
        ),
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
        help="Print progress to stderr every N completed tickers. Use 0 to disable.",
    )
    parser.add_argument(
        "--exchange",
        choices=["nasdaq", "nyse", "amex", "otc"],
        help="Optional WallStreetZen exchange hint. Fastest for Zen because it tries only one exchange.",
    )
    parser.add_argument(
        "--zen-exchanges",
        help=(
            "Comma-separated Zen exchanges to try when --exchange is not set. "
            "Example: --zen-exchanges nasdaq,nyse. Default: nasdaq,nyse,amex,otc."
        ),
    )
    parser.add_argument(
        "--skip-zen",
        action="store_true",
        help="Do not query WallStreetZen. Zen is treated as missing and the usual missing-Zen penalty applies.",
    )
    parser.add_argument(
        "--zen-top-candidates",
        type=int,
        help=(
            "Two-stage mode for large files: first score all tickers without Zen, "
            "then query Zen only for the top N preliminary candidates. "
            "Example: --top 100 --zen-top-candidates 500."
        ),
    )
    args = parser.parse_args()

    if args.top is not None and args.top <= 0:
        print("ERROR: --top must be a positive integer.", file=sys.stderr)
        return 1

    if args.workers <= 0:
        print("ERROR: --workers must be a positive integer.", file=sys.stderr)
        return 1

    if args.workers > 32:
        print("ERROR: --workers above 32 is too aggressive for these public sites.", file=sys.stderr)
        return 1

    if args.zen_workers <= 0:
        print("ERROR: --zen-workers must be a positive integer.", file=sys.stderr)
        return 1

    if args.zen_workers > 4:
        print("ERROR: --zen-workers above 4 is too aggressive for WallStreetZen.", file=sys.stderr)
        return 1

    if args.progress_every < 0:
        print("ERROR: --progress-every cannot be negative.", file=sys.stderr)
        return 1

    if args.zen_top_candidates is not None and args.zen_top_candidates <= 0:
        print("ERROR: --zen-top-candidates must be a positive integer.", file=sys.stderr)
        return 1

    if args.skip_zen and args.zen_top_candidates is not None:
        print("ERROR: Use either --skip-zen or --zen-top-candidates, not both.", file=sys.stderr)
        return 1

    try:
        zen_exchanges = parse_zen_exchange_order(args.zen_exchanges)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
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
        tickers, top_n = parse_mode(args.items, args.top, args.file_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not tickers:
        print("ERROR: No tickers were entered.", file=sys.stderr)
        return 1

    if args.zen_top_candidates is not None:
        rough_results, errors = score_many(
            tickers,
            args.exchange,
            zen_exchanges,
            args.workers,
            args.progress_every,
            skip_zen=True,
        )
        candidate_count = min(args.zen_top_candidates, len(rough_results))
        candidate_tickers = [result.ticker for result in rough_results[:candidate_count]]
        print(
            f"Zen two-stage mode: querying WallStreetZen only for top "
            f"{candidate_count}/{len(tickers)} preliminary candidates.",
            file=sys.stderr,
        )
        results, second_pass_errors = score_many(
            candidate_tickers,
            args.exchange,
            zen_exchanges,
            args.zen_workers,
            args.progress_every,
            skip_zen=False,
        )
        errors.extend(second_pass_errors)
    else:
        results, errors = score_many(
            tickers,
            args.exchange,
            zen_exchanges,
            args.workers,
            args.progress_every,
            skip_zen=args.skip_zen,
        )

    if top_n is not None:
        results = results[:top_n]

    for result in results:
        print(result_line(result))

    csv_path = write_scores_csv(results)
    print(f"Saved CSV: {csv_path}", file=sys.stderr)

    # Keep failures on stderr so stdout stays clean and copyable.
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
