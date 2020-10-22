import csv
import logging
import multiprocessing as mp
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from .generator import Tweet
from .predictor import TweetPredictor, TweetVectorizer, get_token_mapping

logger = logging.getLogger(__name__)

TweetPrediction = Tuple[Tweet, float]
if TYPE_CHECKING:
    TweetQueue = mp.Queue[Optional[Tweet]]
    PredictedTweetQueue = mp.Queue[Optional[TweetPrediction]]
else:
    TweetQueue = PredictedTweetQueue = mp.Queue

X = TypeVar("X")
Y = TypeVar("Y")


class ConsumerProducer(Generic[X, Y]):
    def __init__(
        self,
        input_queue: "mp.Queue[Optional[X]]",
        output_queue: "Optional[mp.Queue[Optional[Y]]]" = None,
        *,
        name: Optional[str] = None,
        daemon: Optional[bool] = None,
    ):
        self._process = mp.Process(target=self._run, name=name, daemon=daemon)
        self._input_queue = input_queue
        self._output_queue = output_queue

    def start(self) -> None:
        self._process.start()

    def stop(self, block: bool = True) -> None:
        self._input_queue.put(None)
        if block:
            self.join()

    def join(self) -> None:
        self._process.join()

    def __enter__(self) -> None:
        logger.info("Entering %s", self)

    def __exit__(self, *exc: object) -> None:
        logger.info("Exiting %s", self)

    def _run(self) -> None:
        consume = self._consume
        produce = self._output_queue.put if self._output_queue is not None else None
        with self:
            for input in iter(self._input_queue.get, None):
                outputs = consume(input)
                if outputs is not None and produce is not None:
                    for output in outputs:
                        produce(output)

    def _consume(self, item: X) -> Optional[Iterable[Y]]:
        pass


class TweetPredictorConsumerProducer(ConsumerProducer[Tweet, TweetPrediction]):
    def __init__(
        self,
        input_queue: TweetQueue,
        output_queue: PredictedTweetQueue,
        model_file: str,
        vocabulary_file: str,
        batch_size: int = 1,
        name: Optional[str] = None,
        daemon: Optional[bool] = None,
    ):
        super().__init__(input_queue, output_queue, name=name, daemon=daemon)
        self._model_file = model_file
        self._vocabulary_file = vocabulary_file
        self._batch_size = batch_size
        self._batch: List[Tweet] = []

    def __enter__(self) -> None:
        super().__enter__()
        vectorizer = TweetVectorizer(get_token_mapping(self._vocabulary_file))
        # GPU cannot be (automatically) used among several processes
        self._predictor = TweetPredictor(self._model_file, vectorizer, disable_gpu=True)

    def __exit__(self, *exc: object) -> None:
        if self._batch and self._output_queue is not None:
            for tweetpred in self._batch_predict():
                self._output_queue.put(tweetpred)
            self._batch.clear()
        return super().__exit__(*exc)

    def _consume(self, tweet: Tweet) -> Iterable[TweetPrediction]:
        if self._batch_size > 1:
            if len(self._batch) == self._batch_size:
                yield from self._batch_predict()
                self._batch.clear()
            self._batch.append(tweet)
        else:
            prediction = self._predictor.predict(tweet.full_text)
            logger.debug("Predicted tweet %s", tweet.id)
            yield tweet, prediction

    def _batch_predict(self) -> Iterable[TweetPrediction]:
        tweets = self._batch
        predictions = self._predictor.batch_predict(
            [tweet.full_text for tweet in tweets]
        )
        assert len(predictions) == len(tweets)
        ids = [tweet.id for tweet in tweets]
        logger.debug("Predicted batch of %d tweets: %s", len(tweets), ids)
        return zip(tweets, predictions)


class CsvWriterConsumer(ConsumerProducer[TweetPrediction, None]):
    def __init__(
        self,
        queue: PredictedTweetQueue,
        csv_file: str,
        dialect: Union[csv.Dialect, str] = "excel-tab",
        name: Optional[str] = None,
        daemon: Optional[bool] = None,
    ):
        super().__init__(queue, name=name, daemon=daemon)
        self._csv_file = csv_file
        self._dialect = dialect

    def __enter__(self) -> None:
        super().__enter__()
        self._consumed = 0
        self._file = open(self._csv_file, "w", newline="")
        self._writerow = csv.writer(self._file, dialect=self._dialect).writerow
        self._writerow(("tweet_id", "created_at", "full_text", "sentiment"))

    def __exit__(self, *exc: object) -> None:
        self._file.close()
        return super().__exit__(*exc)

    def _consume(self, tweet_prediction: TweetPrediction) -> None:
        tweet, prediction = tweet_prediction
        self._writerow((*tweet, prediction))
        self._consumed += 1
        logger.debug("Wrote row %d for tweet %s", self._consumed, tweet.id)


def multiprocess_tweets(
    tweets: Iterable[Tweet],
    model_file: str,
    vocabulary_file: str,
    output_file: str,
    batch_size: int,
    num_processes: int,
) -> None:
    input_queue: TweetQueue = mp.Queue()
    output_queue: PredictedTweetQueue = mp.Queue()

    # start the predictor workers
    predictors = [
        TweetPredictorConsumerProducer(
            input_queue=input_queue,
            output_queue=output_queue,
            model_file=model_file,
            vocabulary_file=vocabulary_file,
            batch_size=batch_size,
            name=f"PredictorProcess-{i}",
        )
        for i in range(num_processes)
    ]
    for predictor in predictors:
        predictor.start()

    # start the csv writer worker
    writer = CsvWriterConsumer(output_queue, output_file, name="WriterProcess")
    writer.start()

    # push all tweets to the input queue
    for tweet in tweets:
        input_queue.put(tweet)

    # notify all predictor workers to stop
    for predictor in predictors:
        predictor.stop(block=False)

    # block until all predictor workers exit
    for predictor in predictors:
        predictor.join()

    # notify the writer worker to stop
    writer.stop()
