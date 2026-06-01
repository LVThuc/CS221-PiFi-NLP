from pathlib import Path
import csv
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LOG_DIR = Path("logs/rerun_all")
MANIFEST = LOG_DIR / "manifest.csv"
OUT_ALL_CSV = LOG_DIR / "all_results_global.csv"
OUT_ALL_MD = LOG_DIR / "all_results_global.md"
OUT_SELECTED_CSV = LOG_DIR / "selected_by_val_global.csv"
OUT_SELECTED_MD = LOG_DIR / "selected_by_val_global.md"

def parse_best_valid(train_log: str):
    path = Path(train_log)
    if not path.exists():
        return None, None

    text = path.read_text(errors="ignore")
    matches = re.findall(r"Best valid at epoch\s+(\d+)\s+-\s+accuracy:\s+([0-9.]+)", text)
    if not matches:
        return None, None

    ep, acc = matches[-1]
    return int(ep), float(acc)

def parse_test_loss(test_log: str):
    path = Path(test_log)
    if not path.exists():
        return None

    text = path.read_text(errors="ignore")
    m = re.search(r"Done!\s+-\s+TEST\s+-\s+Loss:\s+([0-9.]+)\s+-\s+Acc:\s+([0-9.]+)\s+-\s+F1:\s+([0-9.]+)", text)
    if not m:
        return None
    return float(m.group(1))

def global_metrics(pred_file: str):
    path = Path(pred_file)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if not {"gold", "pred"}.issubset(df.columns):
        return None

    y = df["gold"]
    p = df["pred"]

    return {
        "Global Acc": accuracy_score(y, p),
        "Global Macro-F1": f1_score(y, p, average="macro", zero_division=0),
        "Global Weighted-F1": f1_score(y, p, average="weighted", zero_division=0),
    }

def fmt(x):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

def write_md(df: pd.DataFrame, path: Path):
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines))

rows = []

with MANIFEST.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        best_ep, best_val = parse_best_valid(r["train_log"])
        test_loss = parse_test_loss(r["test_log"])
        metrics = global_metrics(r["pred_file"])

        row = {
            "Dataset": r["dataset"],
            "Encoder": r["encoder"],
            "Method": r["method"],
            "LLM": r["llm"],
            "LR": r["lr"],
            "Epochs": int(r["num_epochs"]),
            "Best Val Epoch": best_ep,
            "Best Val Acc": best_val,
            "Test Loss": test_loss,
            "Global Acc": None,
            "Global Macro-F1": None,
            "Global Weighted-F1": None,
            "Pred File": r["pred_file"],
        }

        if metrics is not None:
            row.update(metrics)

        rows.append(row)

df = pd.DataFrame(rows)

sort_cols = ["Dataset", "Encoder", "Method", "LLM", "LR"]
df = df.sort_values(sort_cols).reset_index(drop=True)

df.to_csv(OUT_ALL_CSV, index=False)
write_md(df, OUT_ALL_MD)

# Select best LR/config by validation inside each dataset/encoder/method/llm group.
# This is mainly for imdb+mbert where LR sweep exists.
selected = []
group_cols = ["Dataset", "Encoder", "Method", "LLM"]

for _, g in df.groupby(group_cols, dropna=False):
    gg = g.copy()
    gg["_val"] = gg["Best Val Acc"].fillna(-1)
    gg = gg.sort_values(["_val", "Global Macro-F1", "Global Acc"], ascending=[False, False, False])
    selected.append(gg.iloc[0].drop(labels=["_val"]))

sel = pd.DataFrame(selected)
sel = sel.sort_values(sort_cols).reset_index(drop=True)

sel.to_csv(OUT_SELECTED_CSV, index=False)
write_md(sel, OUT_SELECTED_MD)

print("\n=== ALL RESULTS ===")
print(OUT_ALL_MD.read_text())

print("\n=== SELECTED BY VALIDATION ===")
print(OUT_SELECTED_MD.read_text())

print(f"\nSaved:")
print(f"- {OUT_ALL_CSV}")
print(f"- {OUT_ALL_MD}")
print(f"- {OUT_SELECTED_CSV}")
print(f"- {OUT_SELECTED_MD}")
