# CS221 PiFi NLP Classification

## Group 1

- Lê Văn Thức 24521748
- Bảo Quý Định Tân 24520028
- Lê Phạm Thành Nhân 24520022

![pifi_figure](https://github.com/user-attachments/assets/e73cbce8-e680-419e-a883-13d05c5e2d98)

This repository contains our reproduction and extension of **PiFi: Plug-in and Fine-tuning**, focusing on text classification with BERT/mBERT and Qwen2 plug-in layers.

Original PiFi repository: <https://github.com/khyun8072/PiFi>

We evaluate:

- **IMDB**: English movie-review sentiment classification
- **UIT-VSFC**: Vietnamese student-feedback sentiment classification
- **ChnSentiCorp**: Chinese hotel-review sentiment classification
- **zh_sentiment**: Chinese 3-class sentiment classification

Models:

- Base fine-tuning: BERT, mBERT
- PiFi: BERT/mBERT + one frozen Qwen2 Transformer layer
  - Qwen2-0.5B
  - Qwen2-1.5B
  - Qwen2-7B

Metrics:

- Accuracy
- Global Macro-F1

Important: final F1 is recomputed from saved predictions. We do not rely only on the original batch-level F1 logging.

---

## 1. Get the repository

### Option A: Clone from GitHub

```bash
git clone git@github.com:LVThuc/CS221-PiFi-NLP.git
cd CS221-PiFi-NLP
```

If the repository already exists locally, update it first:

```bash
cd CS221-PiFi-NLP
git pull origin main
```

Then enter the classification module:

```bash
cd Classification
```

### Option B: Initialize Git from a copied folder

If this folder was copied manually and is not a Git repository yet:

```bash
cd CS221-PiFi-NLP
git init
git remote add origin git@github.com:LVThuc/CS221-PiFi-NLP.git
git pull origin main
cd Classification
```

If `git pull` reports divergent branches, use:

```bash
git pull --rebase origin main
```

---

## 2. Create the Python environment

This project uses **uv** and `pyproject.toml` to manage dependencies.

### Recommended: using uv

Install `uv` if it is not available:

```bash
pip install uv
```

From the repository root, sync the environment:

```bash
uv sync
```

Activate the uv virtual environment:

```bash
source .venv/bin/activate
```

Then enter the classification module:

```bash
cd Classification
```

After activation, run commands normally, for example:

```bash
./run_all_global_metrics.sh
```

### Alternative: using Conda or normal pip

If `uv` is not available, you can create a Python environment manually:

```bash
conda create -n cs221-pifi python=3.12 -y
conda activate cs221-pifi
```

Then install the project from `pyproject.toml`:

```bash
pip install -U pip
pip install -e .
```

If some packages are still missing, install the common dependencies manually:

```bash
pip install torch transformers datasets scikit-learn pandas numpy tqdm beautifulsoup4 lxml tensorboard wandb
```

In this mode, remove `uv run` from the beginning of commands and run them directly after activating the environment.

Optional Hugging Face login:

```bash
huggingface-cli login
```

This is useful if the server has no existing Hugging Face token or if model download requires authentication.

---

## 3. Prepare ChnSentiCorp local data

IMDB, UIT-VSFC, and zh_sentiment are loaded from Hugging Face.

ChnSentiCorp is prepared locally because some Hugging Face versions point to broken Google Drive links. Run this once from the `Classification/` directory:

```bash
mkdir -p data/chnsenticorp

python - <<'PY'
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"

out_dir = Path("data/chnsenticorp")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(url)

text_col = "review"
label_col = "label"

df = df[[text_col, label_col]].dropna()
df = df.rename(columns={text_col: "text", label_col: "label"})
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)
df = df[df["label"].isin([0, 1])].reset_index(drop=True)

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"],
)

train_df.to_csv(out_dir / "train.csv", index=False)
valid_df.to_csv(out_dir / "validation.csv", index=False)
test_df.to_csv(out_dir / "test.csv", index=False)

print("Saved ChnSentiCorp splits:")
for name, part in [("train", train_df), ("validation", valid_df), ("test", test_df)]:
    print(name, part.shape, part["label"].value_counts().to_dict())
PY
```

Check the generated files:

```bash
ls -lh data/chnsenticorp
```

Expected files:

```text
train.csv
validation.csv
test.csv
```

---

## 4. Run all experiments

The main script is:

```bash
./run_all_global_metrics.sh
```

Recommended full command:

```bash
DATASETS_STR="imdb uit_vsfc chnsenticorp zh_sentiment" \
ENCODERS_STR="bert mbert" \
NUM_EPOCHS=3 \
KEEP_WEIGHTS=0 \
SKIP_EXISTING=0 \
./run_all_global_metrics.sh
```

Meaning of the variables:

- `DATASETS_STR`: datasets to run
- `ENCODERS_STR`: encoders to test
- `NUM_EPOCHS=3`: train each model for 3 epochs
- `KEEP_WEIGHTS=0`: delete model/checkpoint weights after each test to save disk
- `SKIP_EXISTING=0`: rerun experiments from scratch

The script runs:

- Base fine-tuning
- PiFi with Qwen2-0.5B
- PiFi with Qwen2-1.5B
- PiFi with Qwen2-7B

For **IMDB + mBERT**, the script tests both learning rates:

```text
2e-5
5e-5
```

The final selected report chooses the best learning-rate run by validation accuracy, not by test performance.

---

## 5. Generate the result tables

After all experiments finish, run:

```bash
python collect_global_results.py
```

This creates:

```text
logs/rerun_all/all_results_global.csv
logs/rerun_all/all_results_global.md
logs/rerun_all/selected_by_val_global.csv
logs/rerun_all/selected_by_val_global.md
```

View the full result table:

```bash
cat logs/rerun_all/all_results_global.md
```

View the selected result table:

```bash
cat logs/rerun_all/selected_by_val_global.md
```

The selected table is the main table used in the report. It selects repeated learning-rate runs by validation accuracy.

---

## 6. Expected selected results

Exact values may vary slightly depending on hardware, library versions, and random state. Our selected results include the following.

### IMDB

| Dataset | Encoder | Method | LLM | Acc | Macro-F1 |
|---|---|---|---|---:|---:|
| IMDB | BERT | Base | -- | 0.8696 | 0.8695 |
| IMDB | BERT | PiFi | Qwen2-1.5B | 0.8678 | 0.8677 |
| IMDB | mBERT | Base | -- | 0.8400 | 0.8400 |
| IMDB | mBERT | PiFi | Qwen2-0.5B | 0.8427 | 0.8426 |
| IMDB | mBERT | PiFi | Qwen2-7B | 0.8428 | 0.8426 |

### UIT-VSFC

| Dataset | Encoder | Method | LLM | Acc | Macro-F1 |
|---|---|---|---|---:|---:|
| UIT-VSFC | BERT | Base | -- | 0.8973 | 0.7071 |
| UIT-VSFC | BERT | PiFi | Qwen2-7B | 0.8989 | 0.7393 |
| UIT-VSFC | mBERT | Base | -- | 0.9182 | 0.7692 |
| UIT-VSFC | mBERT | PiFi | Qwen2-0.5B | 0.9128 | 0.7724 |

### Chinese robustness checks

| Dataset | Encoder | Method | LLM | Acc | Macro-F1 |
|---|---|---|---|---:|---:|
| ChnSentiCorp | mBERT | Base | -- | 0.9112 | 0.8992 |
| ChnSentiCorp | mBERT | PiFi | Qwen2-1.5B | 0.8958 | 0.8841 |
| zh_sentiment | mBERT | Base | -- | 0.7947 | 0.7954 |
| zh_sentiment | mBERT | PiFi | Qwen2-0.5B | 0.7790 | 0.7785 |

---

## 7. Reproduce a smaller subset

To run only UIT-VSFC with mBERT:

```bash
DATASETS_STR="uit_vsfc" \
ENCODERS_STR="mbert" \
NUM_EPOCHS=3 \
KEEP_WEIGHTS=0 \
SKIP_EXISTING=0 \
./run_all_global_metrics.sh

python collect_global_results.py
cat logs/rerun_all/selected_by_val_global.md
```

To run only Chinese datasets with mBERT:

```bash
DATASETS_STR="chnsenticorp zh_sentiment" \
ENCODERS_STR="mbert" \
NUM_EPOCHS=3 \
KEEP_WEIGHTS=0 \
SKIP_EXISTING=0 \
./run_all_global_metrics.sh

python collect_global_results.py
cat logs/rerun_all/selected_by_val_global.md
```
