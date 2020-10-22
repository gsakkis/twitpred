# twitpred

Twitpred is a small library for performing sentiment analysis on Twitter statuses (tweets). It provides a Python API and a command line interface (CLI) for collecting tweets and assigning to each tweet a sentiment score predicted by a trained neural network model.

## Installation

Twitpred can be installed from source by cloning the [Git](https://github.com/gsakkis/twitpred) repository:

      git clone https://github.com/gsakkis/twitpred.git
      cd twitpred
      python setup.py install

## Command Line Interface

Once the package is installed, it also installs a CLI called `twitpred`. It accepts several optional arguments and four required:

- `-c/--credentials`: the Twitter API credentials YAML file
- `-m/--model`: the trained TensorFlow Keras model saved as HDF5 file
- `-v/--vocabulary`: the JSON file containing the word-to-index vocabulary
- `-o/--output`: the output CSV file

Here's a sample run for collecting and scoring 1000 tweets in English that contain at least one of the hashtags `#Trump`, `#Biden`, `#elections`, `#trump`, `#biden`, scored by 4 worker processes in batches of 10 at a time and saved as `output.csv`:

```
$ twitpred -c twitter_credentials.yaml \
           -m model.h5 \
           -v w2i.json \
           -o output.csv \
           --limit=1000 \
           --lang=en \
           --batch=10 \
           --processes=4 \
           --loglevel=DEBUG \
           '#Trump' '#Biden' '#elections' '#trump' '#biden'
```

To see the full list of options run `twitpred --help`:

```
$ twitpred --help

usage: twitpred [-h] [--lang LANG] [--limit LIMIT] [--streaming] [--batch B]
                [--processes PROCESSES]
                [--loglevel {DEBUG,INFO,WARNING,ERROR}] -c CREDENTIALS -m
                MODEL -v VOCABULARY -o OUTPUT
                terms [terms ...]

Perform sentiment analysis on tweets

positional arguments:
  terms                 Terms or phrases to match in tweets

optional arguments:
  -h, --help            show this help message and exit
  --lang LANG           ISO 639-1 code of the language to restrict tweets
                        (default: None)
  --limit LIMIT         Maximum number of tweets to process, or unlimited if
                        unspecified (default: None)
  --streaming           Use the Twitter streaming API to fetch realtime tweets
                        instead of the search API (default: False)
  --batch B             Score batches of B tweets at a time (default: 1)
  --processes PROCESSES
                        Number of predictor processes to run in parallel
                        (default: 1)
  --loglevel {DEBUG,INFO,WARNING,ERROR}
                        Logging level (default: INFO)

required arguments:
  -c CREDENTIALS, --credentials CREDENTIALS
                        Twitter API credentials YAML file (default: None)
  -m MODEL, --model MODEL
                        TensorFlow HDF5 file (default: None)
  -v VOCABULARY, --vocabulary VOCABULARY
                        Word-to-index vocabulary JSON file (default: None)
  -o OUTPUT, --output OUTPUT
                        Output CSV file (default: None)
```

## Code Organization

The code has been structured with modularity, extensibility and maintainability in mind. It is contained in a `twitpred` package that consists of the following modules:

### generator

Generates tweets by accessing the Twitter API.

- `Tweet`: namedtuple with the attributes of interest (`id`, `created_at`, `full_text`) along with a converter from `tweepy.Status` instances.
- `get_twitter_api`: function for creating a `tweepy.API` instance from the credentials in a YAML file.
- `generate_tweets`: generator function that returns a (finite or infinite) iterator of tweets by accessing the Twitter API. Both the [Search](https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets) and the [realtime Streaming](https://developer.twitter.com/en/docs/twitter-api/v1/tweets/filter-realtime/overview) API are supported transparently to the user.

### predictor

Provides text preprocessing, tokenization, vectorization and sentiment prediction.

- `get_token_mapping`: function for reading a word-to-index mapping from a JSON file.
- `TweetVectorizer`: callable class responsible for tokenizing text and mapping it to a vector (sequence of integers) using a word-to-index mapping.
- `TweetPredictor`: class that takes a trained TensorFlow model and a `TweetVectorizer` and uses them to predict tweet sentiment scores. Both single and batch predictions are supported.

### processor

Implements the overall tweet processing logic.

- `process_tweets`: creates a `TweetVectorizer` and a `TweetPredictor`, uses them to compute predictions on an iterable of tweets (either one tweet at a time or in batches) and writes the tweets along with the predictions to the output csv file.
- `iter_predictions`: generates `(tweet, prediction)` pairs from an iterable of tweets, scoring one tweet at a time.
- `iter_batch_predictions`: generates `(tweet, prediction)` pairs from an iterable of tweets, scoring one batch of tweets at a time.
- `iter_batches`: generic function that splits an arbitrary iterable into batches.

### multiprocessor

Concurrent version of the `processor`, implemented as a multiprocessing producer-consumer pattern.

- `ConsumerProducer`: generic base class with a [Process](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process)-like API that encapsulates a consumer of a [multiprocessing.Queue](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue) that may (optionally) also act as a producer for another queue.
- `TweetPredictorConsumerProducer`: `ConsumerProducer` subclass that consumes tweets from an input queue, makes predictions using a `TweetPredictor` and produces `(tweet, prediction)` pairs to an output queue.
- `CsvWriterConsumer`: `ConsumerProducer` subclass that consumes `(tweet, prediction)` pairs from a queue and writes them to a csv file.
- `multiprocess_tweets`: function that drives the whole pipeline:
  - Starts `TweetPredictorConsumerProducer` and `CsvWriterConsumer` worker processes.
  - Feeds tweets into the input queue to be picked up by the `TweetPredictorConsumerProducer` workers.
  - Stops the workers after all messages are picked from each queue.

### cli

The command line interface entry point.

- Defines the argument parser.
- Configures logging.
- Parses the cli arguments.
- Invokes the tweet [generator](#generator) and passes it to the tweet [processor](#processor) or [multiprocessor](#multiprocessor).

## Code Quality

To maintain high code quality, the project employs:
- Type annotations on all function and method signatures.
- [pre-commit](https://pre-commit.com/) hooks:
  - [black](https://github.com/ambv/black) for consistent formatting.
  - [isort](https://github.com/PyCQA/isort) for consistent import order.
  - [flake8](https://gitlab.com/pycqa/flake8) for code style and quality checking.
  - [mypy](https://github.com/python/mypy) for optional static typing checking.

## To Do

Due to lack of time, the following are currently missing:

- Handling Twitter API rate limiting and overall error handling.
- Unit & integration tests.
- Code documentation (docstrings).
