import csv
import itertools as it
import logging
from typing import Iterable, List, Tuple, TypeVar

from .generator import Tweet
from .predictor import TweetPredictor, TweetVectorizer, get_token_mapping

logger = logging.getLogger(__name__)


def process_tweets(
    tweets: Iterable[Tweet],
    model_file: str,
    vocabulary_file: str,
    output_file: str,
    batch_size: int = 1,
) -> None:
    vectorizer = TweetVectorizer(get_token_mapping(vocabulary_file))
    predictor = TweetPredictor(model_file, vectorizer)
    if batch_size > 1:
        tweet_preds = iter_batch_predictions(tweets, predictor, batch_size)
    else:
        tweet_preds = iter_predictions(tweets, predictor)
    with open(output_file, "w", newline="") as f:
        writerow = csv.writer(f, dialect="excel-tab").writerow
        writerow(("tweet_id", "created_at", "full_text", "sentiment"))
        for i, (tweet, prediction) in enumerate(tweet_preds, start=1):
            writerow((*tweet, prediction))
            logger.debug("Wrote row %d for tweet %s", i, tweet.id)


def iter_predictions(
    tweets: Iterable[Tweet], predictor: TweetPredictor
) -> Iterable[Tuple[Tweet, float]]:
    for tweet in tweets:
        prediction = predictor.predict(tweet.full_text)
        logger.debug("Predicted tweet %s", tweet.id)
        yield tweet, prediction


def iter_batch_predictions(
    tweets: Iterable[Tweet], predictor: TweetPredictor, batch_size: int
) -> Iterable[Tuple[Tweet, float]]:
    for batch in iter_batches(tweets, batch_size):
        predictions = predictor.batch_predict([tweet.full_text for tweet in batch])
        assert len(predictions) == len(batch)
        ids = [tweet.id for tweet in batch]
        logger.debug("Predicted batch of %d tweets: %s", len(batch), ids)
        yield from zip(batch, predictions)


T = TypeVar("T")


def iter_batches(iterable: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    """Split `iterable` into lists of size `batch_size, except for the last batch which can be shorter.

    >>> list(iter_chunks('ABCDEFG', 3))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    iterator = iter(iterable)
    return iter(lambda: list(it.islice(iterator, batch_size)), [])
