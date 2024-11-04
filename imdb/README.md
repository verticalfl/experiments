# Sentiment Analysis on IMDB dataset

In this setting, one party has the reviews and the other party has the labels.
The party with the labels is helping the party with the reviews train a model
without sharing the labels themselves.

These examples require additional pip packages.

```bash
pip install tensorflow_hub tensorflow_datasets
```

Note: At this time, only the single-process setting is supported (--party=b, the
default if no flag is passed).
