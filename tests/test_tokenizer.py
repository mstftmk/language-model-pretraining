import random
from collections import Counter

import pytest

from tinylm.tokenizer import BPETokenizer, _merge, count_pretokens, learn_merges

EOT = "<|endoftext|>"

CORPUS = (
    "Once upon a time there was a little cat. The cat liked to play in the sun."
    f"{EOT}"
    "One day, the little dog saw the cat. \"Hello, cat!\" said the dog. They played all day."
    f"{EOT}"
    "Lily had a red ball. She threw the ball to her friend Tom, and Tom threw it back."
) * 20


def naive_merges(word_counts: Counter, num_merges: int) -> list[tuple[int, int]]:
    """Reference trainer: recount every pair after every merge (slow, obvious)."""
    words = {tuple(w): c for w, c in word_counts.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += c
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))  # same tie-break as learn_merges
        new_id = 256 + len(merges)
        merges.append(best)
        merged = Counter()
        for w, c in words.items():
            merged[tuple(_merge(list(w), best, new_id))] += c
        words = merged
    return merges


def naive_encode_pretoken(tok: BPETokenizer, token: bytes) -> list[int]:
    """Reference encoder: apply every merge in training order."""
    ids = list(token)
    for r, pair in enumerate(tok.merges):
        ids = _merge(ids, pair, 256 + r)
    return ids


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=300)


def test_merge_is_left_to_right_and_non_overlapping():
    assert _merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert _merge([1, 2, 1, 2], (1, 2), 9) == [9, 9]
    assert _merge([], (1, 2), 9) == []


def test_incremental_trainer_matches_naive_trainer():
    counts = count_pretokens(CORPUS)
    assert learn_merges(counts, 60) == naive_merges(counts, 60)


def test_incremental_trainer_matches_naive_on_random_text():
    rng = random.Random(0)
    alphabet = "aab bc  ç ğ ş !."
    text = "".join(rng.choice(alphabet) for _ in range(3000))
    counts = count_pretokens(text)
    assert learn_merges(counts, 80) == naive_merges(counts, 80)


def test_vocab_size_is_exact(tok):
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256 - 1
    assert tok.special_to_id[EOT] == 299


def test_first_merge_is_most_frequent_pair():
    counts = count_pretokens(CORPUS)
    pairs = Counter()
    for w, c in counts.items():
        for p in zip(w, w[1:]):
            pairs[p] += c
    top = max(pairs.values())
    first = learn_merges(counts, 1)[0]
    assert pairs[first] == top


@pytest.mark.parametrize(
    "text",
    [
        "",
        "the cat",
        "  leading and trailing spaces  ",
        "new\nlines\n\n and\ttabs",
        "Türkçe karakterler: ğüşiöç İĞÜŞÖÇ",
        "emoji 🙂🚀 and 漢字",
        "numbers 12345 and punctuation?!...",
        f"two stories{EOT}joined",
    ],
)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_random_unicode(tok):
    rng = random.Random(1)
    for _ in range(50):
        text = "".join(chr(rng.randint(0, 0x2FFF)) for _ in range(rng.randint(0, 40)))
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        assert tok.decode(tok.encode(text)) == text


def test_encode_matches_reference(tok):
    for word in [" the", " cat", " little", "Once", " played", "Hello", " threw"]:
        raw = word.encode("utf-8")
        assert tok._encode_pretoken(raw) == naive_encode_pretoken(tok, raw)


def test_special_token_is_single_id_and_never_split(tok):
    ids = tok.encode(f"a{EOT}b")
    assert ids.count(tok.special_to_id[EOT]) == 1
    # a partial special token is ordinary text
    assert tok.special_to_id[EOT] not in tok.encode("<|endof")


def test_training_ignores_special_tokens():
    counts = count_pretokens(f"hello{EOT}world")
    assert all(b"endoftext" not in w for w in counts)


def test_merges_do_not_cross_pretoken_boundaries(tok):
    # "cat." is two pre-tokens (" cat", "."), so no learned token may hold "t."
    assert all(b"t." not in v for v in tok.vocab.values())


def test_compresses_training_text(tok):
    text = CORPUS[:2000]
    assert len(tok.encode(text)) < 0.6 * len(text.encode("utf-8"))


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=320)
    b = BPETokenizer.train(CORPUS, vocab_size=320)
    assert a.merges == b.merges


def test_save_load_round_trip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_to_id == tok.special_to_id
    text = "Lily and Tom played with the red ball."
    assert loaded.encode(text) == tok.encode(text)


def test_too_small_vocab_raises():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", vocab_size=200)
