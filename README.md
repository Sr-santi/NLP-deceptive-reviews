# Challenge 1

Classify Chicago hotel reviews as `truthful` or `deceptive`. Notebook: `text_classification.ipynb`.

## Data

`../LogisticRegression_SpamOpinion/Datos/deceptive-opinion.csv`

1,600 reviews. 800 truthful, 800 deceptive. 400 positive and 400 negative inside each class.

| Column | Role |
|---|---|
| `deceptive` | Target |
| `text` | Input |
| `polarity` | Not the target |
| `hotel` | 20 Chicago hotels |
| `source` | Not a feature. `MTurk` marks the fake reviews. |

## Setup

```bash
uv sync
```
