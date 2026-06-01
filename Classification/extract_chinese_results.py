from pathlib import Path
import re
import csv

LOG_DIR = Path("logs")
OUT_CSV = LOG_DIR / "chinese_results_summary.csv"
OUT_MD = LOG_DIR / "chinese_results_summary.md"

DATASETS = ["chnsenticorp", "zh_sentiment"]

def restore_llm_name(s: str) -> str:
    if not s or s == "-":
        return "-"
    # log filename used qwen2_0_5b instead of qwen2_0.5b
    s = s.replace("qwen2_0_5b", "qwen2_0.5b")
    s = s.replace("qwen2_1_5b", "qwen2_1.5b")
    return s

def parse_name(path: Path):
    stem = path.stem

    dataset = None
    for d in DATASETS:
        if stem.startswith(d + "_"):
            dataset = d
            rest = stem[len(d) + 1:]
            break

    if dataset is None:
        return None

    # expected:
    # chnsenticorp_mbert_base_train
    # chnsenticorp_mbert_pifi_qwen2_0_5b_train
    # zh_sentiment_mbert_pifi_qwen2_7b_test
    parts = rest.split("_")
    encoder = parts[0]

    if "_base_" in stem:
        method = "base"
        llm = "-"
    elif "_pifi_" in stem:
        method = "pifi"
        m = re.search(r"_pifi_(.+)_(train|test)$", stem)
        llm = restore_llm_name(m.group(1)) if m else "?"
    else:
        return None

    kind = "train" if stem.endswith("_train") else "test" if stem.endswith("_test") else "?"
    return dataset, encoder, method, llm, kind

def parse_train(path: Path):
    text = path.read_text(errors="ignore")
    matches = re.findall(r"Best valid at epoch\s+(\d+)\s+-\s+accuracy:\s+([0-9.]+)", text)
    if not matches:
        return None, None
    ep, acc = matches[-1]
    return int(ep), float(acc)

def parse_test(path: Path):
    text = path.read_text(errors="ignore")
    m = re.search(
        r"Done!\s+-\s+TEST\s+-\s+Loss:\s+([0-9.]+)\s+-\s+Acc:\s+([0-9.]+)\s+-\s+F1:\s+([0-9.]+)",
        text
    )
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

rows = {}

for path in sorted(LOG_DIR.glob("*.log")):
    info = parse_name(path)
    if info is None:
        continue

    dataset, encoder, method, llm, kind = info
    key = (dataset, encoder, method, llm)

    if key not in rows:
        rows[key] = {
            "Dataset": dataset,
            "Encoder": encoder,
            "Method": method,
            "LLM": llm,
            "Best Val Epoch": "",
            "Best Val Acc": "",
            "Test Loss": "",
            "Test Acc": "",
            "Test F1": "",
        }

    if kind == "train":
        ep, acc = parse_train(path)
        if ep is not None:
            rows[key]["Best Val Epoch"] = ep
            rows[key]["Best Val Acc"] = f"{acc:.4f}"

    elif kind == "test":
        loss, acc, f1 = parse_test(path)
        if loss is not None:
            rows[key]["Test Loss"] = f"{loss:.4f}"
            rows[key]["Test Acc"] = f"{acc:.4f}"
            rows[key]["Test F1"] = f"{f1:.4f}"

result = list(rows.values())

# Sort: dataset, encoder, base first, then qwen size
def sort_key(r):
    method_rank = 0 if r["Method"] == "base" else 1
    llm_rank = {"-": 0, "qwen2_0.5b": 1, "qwen2_1.5b": 2, "qwen2_7b": 3}.get(r["LLM"], 9)
    return (r["Dataset"], r["Encoder"], method_rank, llm_rank)

result.sort(key=sort_key)

fields = ["Dataset", "Encoder", "Method", "LLM", "Best Val Epoch", "Best Val Acc", "Test Loss", "Test Acc", "Test F1"]

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(result)

def md_table(rows):
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in fields) + " |")
    return "\n".join(lines)

OUT_MD.write_text(md_table(result))

print(md_table(result))
print()
print(f"Saved CSV to {OUT_CSV}")
print(f"Saved Markdown to {OUT_MD}")
