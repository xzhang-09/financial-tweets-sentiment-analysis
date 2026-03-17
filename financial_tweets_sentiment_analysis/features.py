import re
from typing import List


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
CASHTAG_PATTERN = re.compile(r"\$[A-Za-z][A-Za-z0-9._-]*")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9$@# ]+")


def extract_ticker_mentions(text: str) -> List[str]:
    return sorted({match.upper() for match in CASHTAG_PATTERN.findall(text or "")})


def has_url(text: str) -> bool:
    return bool(URL_PATTERN.search(text or ""))


def clean_tweet_text(text: str) -> str:
    text = (text or "").lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = text.replace("&amp;", " and ")
    text = NON_ALNUM_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def is_short_text(text: str, threshold: int = 8) -> bool:
    return len((text or "").split()) < threshold


def is_news_headline(text: str) -> bool:
    lowered = (text or "").lower()
    keywords = ("breaking", "reports", "upgrades", "downgrades", "earnings", "shares", "stock")
    return any(keyword in lowered for keyword in keywords)
