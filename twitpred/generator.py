import itertools as it
import logging
from collections import namedtuple
from queue import Queue
from typing import Any, Callable, Iterator, Optional, Sequence

import tweepy
import yaml

logger = logging.getLogger(__name__)


class Tweet(namedtuple("_Tweet", ["id", "created_at", "full_text"])):
    __slots__ = ()

    _status_text_getters: Sequence[Callable[[tweepy.Status], str]] = (
        lambda status: status.full_text,
        lambda status: status.extended_tweet["full_text"],
        lambda status: status.text,
    )

    @classmethod
    def from_status(cls, status: tweepy.Status) -> "Tweet":
        for text_getter in cls._status_text_getters:
            try:
                return cls(status.id, status.created_at, text_getter(status))
            except AttributeError:
                pass
        raise AssertionError(f"Cannot determine text from status {status!r}")


def get_twitter_api(credentials_yaml: str) -> tweepy.API:
    with open(credentials_yaml) as f:
        credentials = yaml.safe_load(f)["twitter_credentials"]

    auth = tweepy.OAuthHandler(credentials["api_key"], credentials["api_secret_key"])
    auth.set_access_token(
        credentials.get("access_token"), credentials.get("access_token_secret")
    )
    return tweepy.API(auth)


def generate_tweets(
    api: tweepy.API,
    terms: Sequence[str],
    lang: Optional[str] = None,
    limit: int = 0,
    streaming: bool = False,
) -> Iterator[Tweet]:
    if streaming:
        iter_statuses = StatusStreamGenerator(
            api=api, limit=limit, track=terms, languages=[lang] if lang else None
        )
    else:
        iter_statuses = tweepy.Cursor(
            api.search, q=" OR ".join(terms), lang=lang, tweet_mode="extended"
        ).items(limit)

    for status in iter_statuses:
        # Keep the retweeted status if this status is a retweet
        status = getattr(status, "retweeted_status", status)
        yield Tweet.from_status(status)


class StatusStreamGenerator(tweepy.StreamListener):
    def __init__(self, api: tweepy.API, limit: int = 0, **kwargs: Any):
        super().__init__(api)
        self._range = range(limit) if limit > 0 else it.count()
        self._queue: Queue[tweepy.Status] = Queue()
        self._stream = tweepy.Stream(api.auth, listener=self)
        self._stream.filter(is_async=True, **kwargs)

    def on_status(self, status: tweepy.Status) -> None:
        self._queue.put(status)

    def __iter__(self) -> Iterator[tweepy.Status]:
        get = self._queue.get
        try:
            for _ in self._range:
                yield get()
        finally:
            self._stream.disconnect()
            logger.debug(
                "Stream disconnected: %d queued statuses discarded", self._queue.qsize()
            )
