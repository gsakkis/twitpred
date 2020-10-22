import json
import re
from collections import defaultdict
from typing import Iterable, List, Mapping

import tensorflow as tf
from nltk.tokenize.casual import EMOTICON_RE, URLS, TweetTokenizer

RE_FLAGS = re.VERBOSE | re.IGNORECASE | re.UNICODE

# urls, see https://gist.github.com/winzig/8894715
URLS_RE = re.compile(URLS, RE_FLAGS)
# Twitter username
USERNAMES_RE = re.compile(r"(?:@[\w_]+)", RE_FLAGS)
# Twitter hashtags
HASHTAGS_RE = re.compile(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", RE_FLAGS)
# Numbers including decimals.
NUMBERS_RE = re.compile(
    r"""
(?:
    [+\-]?
    (?:
        \d+[,.]?\d*
        |
        [,.]\d+
    )
)
""",
    RE_FLAGS,
)


def get_token_mapping(vocabulary_file: str) -> Mapping[str, int]:
    with open(vocabulary_file) as f:
        d = json.load(f)
    unknown = d.pop("*#*UNK*#*")
    return defaultdict(lambda: unknown, **d)


class TweetVectorizer:
    _emoticon_mapping = {
        ":-)": "<smileface>",
        ":)": "<smileface>",
        ":D": "<lolface>",
        ":-D": "<lolface>",
        ":|": "<neutralface>",
        ":-(": "<sadface>",
        ":(": "<sadface>",
    }

    def __init__(
        self,
        token_mapping: Mapping[str, int],
        preserve_case: bool = False,
    ):
        self._token_mapping = token_mapping
        self._tokenizer = TweetTokenizer(preserve_case=preserve_case)

    def __call__(self, text: str) -> List[int]:
        token_mapping = self._token_mapping
        return [token_mapping[token] for token in self.tokenize(text)]

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer.tokenize(self.preprocess(text))

    def preprocess(self, text: str) -> str:
        get_emoticon = self._emoticon_mapping.get
        text = URLS_RE.sub("<url>", text)
        text = USERNAMES_RE.sub("<user>", text)
        text = HASHTAGS_RE.sub("<hashtag>", text)
        text = NUMBERS_RE.sub("<number>", text)
        text = EMOTICON_RE.sub(lambda m: get_emoticon(m.group()) or m.group(), text)
        return text


class TweetPredictor:
    def __init__(
        self, model_file: str, vectorizer: TweetVectorizer, disable_gpu: bool = False
    ):
        if disable_gpu:
            tf.config.set_visible_devices([], device_type="GPU")
        self._model = tf.keras.models.load_model(model_file)
        self._vectorizer = vectorizer

    def predict(self, text: str) -> float:
        vector = self._vectorizer(text)
        return self._model.predict([vector]).item()

    def batch_predict(self, texts: Iterable[str]) -> List[float]:
        vectors = list(map(self._vectorizer, texts))
        if not vectors:
            return []
        vectors = tf.keras.preprocessing.sequence.pad_sequences(vectors)
        return self._model.predict(vectors).ravel().tolist()
