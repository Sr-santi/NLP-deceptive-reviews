# Challenge 1

Classify Chicago hotel reviews as truthful or deceptive. The work is in `text_classification.ipynb`.

Target: `deceptive`. Input: `text`. Seed: `28`.

## Data

[Deceptive Opinion Spam Corpus](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus) (Ott et al.), file `deceptive-opinion.csv`.

1,600 reviews. No missing values. 4 review texts are duplicated, so the text column has 1,596 unique values.

| | negative | positive | total |
|---|---:|---:|---:|
| deceptive | 400 | 400 | 800 |
| truthful | 400 | 400 | 800 |
| total | 800 | 800 | 1,600 |

| Column | Role |
|---|---|
| `deceptive` | Target. `truthful` = 0, `deceptive` = 1 |
| `text` | Input |
| `polarity` | Second label. Used to keep the splits balanced, not as a feature |
| `hotel` | 20 Chicago hotels. Context only |
| `source` | Metadata. Not a feature |

`source` identifies the class by itself: all 800 MTurk reviews are deceptive, and the 400 TripAdvisor plus 400 Web reviews are truthful.

## Data comprehension

Reviews are lowercased. Tokens are letters only. Words of 1 or 2 characters are dropped, along with English stopwords and the domain words `hotel`, `hotels`, `room`, `rooms`, `chicago`, `stay`, `stayed`.

A word counts once per review. With a minimum of 15 reviews: 584 words appear in both classes, 134 are mostly deceptive, and 173 are mostly truthful.

The rates below are the share of the 400 reviews in that cell that contain the word. Positive `diff` means the word is more common in deceptive reviews. These words move the same way in positive and negative reviews.

| word | fake, positive | real, positive | fake, negative | real, negative | diff, positive | diff, negative |
|---|---:|---:|---:|---:|---:|---:|
| location | 0.20 | 0.48 | 0.10 | 0.29 | -0.28 | -0.19 |
| floor | 0.03 | 0.19 | 0.10 | 0.20 | -0.16 | -0.11 |
| michigan | 0.08 | 0.22 | 0.01 | 0.07 | -0.15 | -0.06 |
| experience | 0.20 | 0.08 | 0.28 | 0.15 | 0.12 | 0.13 |
| luxury | 0.11 | 0.01 | 0.13 | 0.01 | 0.10 | 0.12 |
| staying | 0.17 | 0.07 | 0.23 | 0.13 | 0.10 | 0.10 |
| recommend | 0.28 | 0.18 | 0.16 | 0.08 | 0.09 | 0.08 |

Deceptive reviews lean on the stay as a story: `experience`, `luxury`, `staying`, `recommend`. Truthful reviews name concrete things: `location`, `floor`, `michigan`, and also `lobby` and `coffee`.

## Shared preparation

The split is stratified on `deceptive` × `polarity`.

| split | reviews | each class × polarity |
|---|---:|---:|
| train | 960 | 240 |
| validation | 320 | 80 |
| test | 320 | 80 |

The vectorizer is fit on the training text only. Same token rule as above, `min_df=5`. That leaves 1,891 terms. Counts are then weighted with TF-IDF, also fit on the training text only.

On the full corpus, before the split, the same vectorizer produces 2,604 terms. That matrix is only for inspection.

## First experiment

A linear classifier in PyTorch Lightning: one `Linear` layer, 1.9K parameters, `BCEWithLogitsLoss`, Adam at learning rate `0.01`, batch size 64. Early stopping watches validation loss with patience 5. The run used the full 50 epochs. The checkpoint is epoch 49.

At the last epoch the training accuracy is 1.000 and the training loss is 0.097. Validation accuracy is 0.847 and validation loss is 0.330. The training set is memorized. The validation score stays below it.

Test set, 320 reviews. Rows are the true class, columns are the prediction.

|  | predicted real | predicted fake |
|---|---:|---:|
| real | 134 | 26 |
| fake | 25 | 135 |

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| real | 0.843 | 0.838 | 0.840 | 160 |
| fake | 0.839 | 0.844 | 0.841 | 160 |
| accuracy |  |  | 0.841 | 320 |

Real and fake are scored almost equally. About 26 real reviews are called fake, and 25 fake reviews are called real.

## Second experiment

Gemini 2.5 Flash (`google/gemini-2.5-flash`) through OpenRouter. The key is read from `.env` and is not stored in the notebook. Each prompt contains only the review text and asks for one word, `truthful` or `deceptive`. Temperature is 0.

This is a behavior check, not a full test-set score. The sample is 40 test reviews: 10 from each class and polarity, drawn with seed 28. The linear model is scored again on these same 40 reviews.

|  | predicted real | predicted fake |
|---|---:|---:|
| linear, real | 17 | 3 |
| linear, fake | 5 | 15 |
| Gemini, real | 16 | 4 |
| Gemini, fake | 11 | 9 |

| model | accuracy | real F1 | fake F1 | fake recall |
|---|---:|---:|---:|---:|
| linear model | 0.800 | 0.810 | 0.789 | 0.750 |
| Gemini 2.5 Flash | 0.625 | 0.681 | 0.545 | 0.450 |

On this sample the linear model remains ahead. Gemini accepts most real reviews (recall 0.800) and misses most fake ones (recall 0.450). Its accuracy here, 0.625, is not comparable to the linear model's 0.841 on the full test set. The matched number for the linear model is 0.800 on these 40 reviews.

## Setup

```bash
uv sync
```

For the second experiment, put the OpenRouter key in `challenge_1/.env`:

```bash
OPENROUTER_API_KEY=your-key
```

That file is gitignored.
