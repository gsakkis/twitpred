"""Perform sentiment analysis on tweets"""

import logging.config
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from .generator import generate_tweets, get_twitter_api
from .multiprocessor import multiprocess_tweets
from .processor import process_tweets


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("terms", nargs="+", help="Terms or phrases to match in tweets")

    # optional arguments
    parser.add_argument(
        "--lang", help="ISO 639-1 code of the language to restrict tweets"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of tweets to process, or unlimited if unspecified",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use the Twitter streaming API to fetch realtime tweets instead of the search API",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="B",
        help="Score batches of B tweets at a time",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of predictor processes to run in parallel",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # required arguments
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-c", "--credentials", required=True, help="Twitter API credentials YAML file"
    )
    required_args.add_argument(
        "-m", "--model", required=True, help="TensorFlow HDF5 file"
    )
    required_args.add_argument(
        "-v", "--vocabulary", required=True, help="Word-to-index vocabulary JSON file"
    )
    required_args.add_argument("-o", "--output", required=True, help="Output CSV file")

    return parser


def config_logging(loglevel: int) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": {
                "standard": {
                    "format": "[%(levelname)s|%(processName)s|%(name)s]: %(message)s"
                },
            },
            "handlers": {
                "default": {
                    "formatter": "standard",
                    "class": "logging.StreamHandler",
                },
            },
            "loggers": {
                "twitpred": {
                    "handlers": ["default"],
                    "level": loglevel,
                },
            },
        }
    )


def main() -> None:
    args = get_parser().parse_args()
    config_logging(getattr(logging, args.loglevel))
    params = dict(
        tweets=generate_tweets(
            api=get_twitter_api(args.credentials),
            terms=args.terms,
            lang=args.lang,
            limit=args.limit,
            streaming=args.streaming,
        ),
        vocabulary_file=args.vocabulary,
        model_file=args.model,
        output_file=args.output,
        batch_size=args.batch,
    )
    if args.processes > 1:
        multiprocess_tweets(num_processes=args.processes, **params)
    else:
        process_tweets(**params)


if __name__ == "__main__":
    main()
