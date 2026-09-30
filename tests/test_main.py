"""Smoke tests for sentence extraction and citation matching."""

from ddgs.exceptions import DDGSException

import main


def test_extract_sentences_keeps_abbreviations_and_drops_digits():
    text = (
        "This is a clean sentence. This has a number 42 in it. "
        "Is this a question? See note [1]. Dr. Smith went home. "
        "The U.S. economy grew. Final fragment"
    )

    assert main.extract_sentences(text) == [
        "This is a clean sentence.",
        "Is this a question?",
        "Dr. Smith went home.",
        "The U.S. economy grew.",
        "Final fragment",
    ]


def test_empty_text_preserves_blank_sentence():
    assert main.extract_sentences("") == [""]


def test_sample_input_loads():
    sentences = main.load_sentences(main.DEFAULT_INPUT)
    assert sentences == ["# Replace this line with the text you want to search for"]


def test_select_citation_picks_highest_score_at_threshold():
    sentence = "The quick brown fox jumps."
    results = [
        {"title": "Low", "href": "https://example.com/low", "body": "unrelated words"},
        {"title": "Exact", "href": "https://example.com/exact", "body": sentence},
    ]

    chosen = main.select_citation(sentence, results)

    assert chosen == {
        "sentence": sentence,
        "url": "https://example.com/exact",
        "title": "Exact",
    }


def test_select_citation_rejects_weak_matches():
    sentence = "The quick brown fox jumps."
    results = [{"title": "Low", "href": "https://example.com/low", "body": "zzzz yyyy"}]

    assert main.select_citation(sentence, results) == {
        "sentence": sentence,
        "url": "original_sentence",
    }
    assert main.select_citation(sentence, None) == {
        "sentence": sentence,
        "url": "original_sentence",
    }


class _FakeDDGS:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def text(self, query, **kwargs):
        backend = kwargs["backend"]
        if backend == "duckduckgo":
            assert query.startswith("inurl:")
            assert kwargs["region"] == "wt-wt"
        elif backend == "auto":
            assert not query.startswith("inurl:")
            assert kwargs["region"] == "us-en"
        else:
            raise AssertionError(backend)
        assert kwargs["max_results"] == main.MAX_RESULTS
        key = (backend, kwargs["page"])
        self.calls.append(key)
        payload = self.pages.get(key, DDGSException("No results found."))
        if isinstance(payload, Exception):
            raise payload
        return payload


def test_search_paginates_and_stops_on_empty_page(monkeypatch):
    fake = _FakeDDGS(
        {
            ("duckduckgo", 1): [{"title": "A", "href": "https://example.com/1", "body": "one"}],
            ("duckduckgo", 2): [{"title": "B", "href": "https://example.com/2", "body": "two"}],
            ("duckduckgo", 3): DDGSException("No results found."),
        }
    )
    monkeypatch.setattr(main, "DDGS", lambda: fake)

    results = main.search_results("A sentence to find.")

    assert fake.calls == [("duckduckgo", 1), ("duckduckgo", 2), ("duckduckgo", 3)]
    assert [item["href"] for item in results] == [
        "https://example.com/1",
        "https://example.com/2",
    ]


def test_search_stops_when_pages_repeat(monkeypatch):
    page = [{"title": "A", "href": "https://example.com/1", "body": "one"}]
    fake = _FakeDDGS({("duckduckgo", 1): page, ("duckduckgo", 2): page})
    monkeypatch.setattr(main, "DDGS", lambda: fake)

    results = main.search_results("A sentence to find.")

    assert fake.calls == [("duckduckgo", 1), ("duckduckgo", 2)]
    assert results == page


def test_search_caps_pages(monkeypatch):
    def page(number):
        return [{"title": "T", "href": f"https://example.com/{number}", "body": "x"}]

    fake = _FakeDDGS(
        {("duckduckgo", number): page(number) for number in range(1, main.MAX_PAGES + 2)}
    )
    monkeypatch.setattr(main, "DDGS", lambda: fake)

    results = main.search_results("A sentence to find.")

    assert fake.calls == [("duckduckgo", number) for number in range(1, main.MAX_PAGES + 1)]
    assert len(results) == main.MAX_PAGES


def test_search_falls_back_when_duckduckgo_has_no_hits(monkeypatch):
    sentence = "A sentence to find."
    fake = _FakeDDGS(
        {("auto", 1): [{"title": "Fallback", "href": "https://example.com/auto", "body": sentence}]}
    )
    monkeypatch.setattr(main, "DDGS", lambda: fake)

    results = main.search_results(sentence)

    assert fake.calls == [("duckduckgo", 1), ("auto", 1), ("auto", 2)]
    assert results == [{"title": "Fallback", "href": "https://example.com/auto", "body": sentence}]


def test_search_reraises_unexpected_client_errors(monkeypatch):
    fake = _FakeDDGS({("duckduckgo", 1): DDGSException("connection failed")})
    monkeypatch.setattr(main, "DDGS", lambda: fake)

    try:
        main.search_results("A sentence to find.")
    except DDGSException as exc:
        assert "connection failed" in str(exc)
    else:
        raise AssertionError("expected DDGSException")
    assert fake.calls == [("duckduckgo", 1)]


def test_process_sentence_uses_collected_pages(monkeypatch):
    sentence = "Cited sentence lives here."
    fake = _FakeDDGS(
        {
            ("duckduckgo", 1): [
                {"title": "Miss", "href": "https://example.com/miss", "body": "nothing"}
            ],
            ("duckduckgo", 2): [
                {"title": "Hit", "href": "https://example.com/hit", "body": sentence}
            ],
        }
    )
    monkeypatch.setattr(main, "DDGS", lambda: fake)

    assert main.process_sentence(sentence) == {
        "sentence": sentence,
        "url": "https://example.com/hit",
        "title": "Hit",
    }


def test_main_prints_matches(tmp_path, monkeypatch, capsys):
    source = tmp_path / "text-to-cite.txt"
    source.write_text("Hello world.")

    monkeypatch.setattr(
        main,
        "process_sentence",
        lambda sentence: {"sentence": sentence, "url": "https://example.com/hello"},
    )

    class FakePool:
        def __init__(self, processes):
            assert processes == 4

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, items):
            return [function(item) for item in items]

    monkeypatch.setattr(main.multiprocessing, "Pool", FakePool)

    main.main(source)

    assert capsys.readouterr().out == "Hello world.: https://example.com/hello\n"
