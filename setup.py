from setuptools import setup

setup(
    name="twitpred",
    version="0.1.0",
    description="Sentiment Analysis on Tweets with Deep Learning",
    url="https://github.com/gsakkis/twitpred",
    author="George Sakkis",
    author_email="george.sakkis@gmail.com",
    packages=["twitpred"],
    install_requires=[
        "nltk==3.5",
        "PyYAML==5.3.1",
        "tensorflow==2.3.1",
        "tweepy==3.8.0",
    ],
    entry_points={"console_scripts": ["twitpred=twitpred.cli:main"]},
    license="MIT",
)
