"""Find likely web sources for sentences in a text file."""

import multiprocessing
import re
from pathlib import Path

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from jaro import jaro_winkler_metric as distance

# Split on sentence-ending punctuation while keeping common abbreviations intact.
SENTENCE_PATTERN = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")

# Drop sentences that contain citation markers or other digits.
NOT_IN_SENTENCE = re.compile(r"\[\d*\]|\d+")

MATCH_THRESHOLD = 0.65
MAX_RESULTS = 100
# ddgs fetches one result page per call. Cap follow-up pages so a query
# still considers up to MAX_RESULTS hits without looping forever.
MAX_PAGES = 10
DEFAULT_INPUT = Path("./text-to-cite.txt")


def extract_sentences(text):
    sentences = []
    for sentence in SENTENCE_PATTERN.split(text):
        if len(NOT_IN_SENTENCE.findall(sentence.strip())) == 0:
            sentences.append(sentence.strip())
    return sentences


def load_sentences(path=DEFAULT_INPUT):
    with open(path) as handle:
        text = handle.read()
    return extract_sentences(text)


def select_citation(sentence, results):
    if not results:
        return {"sentence": sentence, "url": "original_sentence"}

    max_score = 0.0
    max_link = None
    for link in results:
        score = distance(link["body"], sentence)
        if score > max_score:
            max_score = score
            max_link = link

    if max_link and max_score >= MATCH_THRESHOLD:
        return {
            "sentence": sentence,
            "url": max_link["href"],
            "title": max_link["title"],
        }
    return {"sentence": sentence, "url": "original_sentence"}


def _is_empty_search(exc):
    # duckduckgo_search.ddg() returned a false value when a query had no hits.
    # ddgs raises DDGSException("No results found.") for that case.
    return "No results found" in str(exc)


def _collect_pages(client, query, backend, region):
    collected = []
    seen = set()
    for page in range(1, MAX_PAGES + 1):
        try:
            batch = client.text(
                query,
                region=region,
                max_results=MAX_RESULTS,
                page=page,
                backend=backend,
            )
        except DDGSException as exc:
            if _is_empty_search(exc) or collected:
                break
            raise
        if not batch:
            break

        new_count = 0
        for item in batch:
            href = item.get("href")
            if href in seen:
                continue
            seen.add(href)
            collected.append(item)
            new_count += 1
            if len(collected) >= MAX_RESULTS:
                return collected
        if new_count == 0:
            break
    return collected


def search_results(sentence):
    """Search for pages that match a sentence.

    The original script queried DuckDuckGo with an ``inurl:`` operator.
    That endpoint currently answers HTTP 202 without result HTML, and other
    engines in ddgs do not treat ``inurl:`` as a search operator. When the
    DuckDuckGo query has no hits, fall back to a plain-text metasearch.
    """
    client = DDGS()
    collected = _collect_pages(client, "inurl:" + sentence, "duckduckgo", "wt-wt")
    if collected:
        return collected
    return _collect_pages(client, sentence, "auto", "us-en") or None


def process_sentence(sentence):
    return select_citation(sentence, search_results(sentence))


def main(path=DEFAULT_INPUT):
    sentences = load_sentences(path)
    with multiprocessing.Pool(processes=4) as pool:
        sentences_urls = pool.map(process_sentence, sentences)

    for sentence_url in sentences_urls:
        print(f"{sentence_url['sentence']}: {sentence_url['url']}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
