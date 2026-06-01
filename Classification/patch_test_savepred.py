from pathlib import Path
import re
from datetime import datetime

p = Path("task/classification/test.py")
s = p.read_text()

backup = Path(f"task/classification/test.py.bak_savepred_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
backup.write_text(s)
print(f"Backup saved to {backup}")

# imports
if "import os\n" not in s:
    s = s.replace("import argparse\n", "import argparse\nimport os\n", 1)

if "import pandas as pd\n" not in s:
    s = s.replace(
        "from sklearn.metrics import f1_score\n",
        "from sklearn.metrics import f1_score\nimport pandas as pd\n",
        1
    )

# init buffers after model.eval()
if not re.search(r"^\s*all_preds\s*=\s*\[\]", s, flags=re.M):
    lines = s.splitlines()
    new_lines = []
    inserted = False

    for line in lines:
        new_lines.append(line)
        if (not inserted) and "model.eval()" in line:
            indent = re.match(r"^(\s*)", line).group(1)
            new_lines.append("")
            new_lines.append(indent + "all_preds = []")
            new_lines.append(indent + "all_labels = []")
            inserted = True

    if not inserted:
        new_lines = []
        for line in lines:
            if (not inserted) and "with torch.no_grad()" in line:
                indent = re.match(r"^(\s*)", line).group(1)
                new_lines.append(indent + "all_preds = []")
                new_lines.append(indent + "all_labels = []")
                new_lines.append("")
                inserted = True
            new_lines.append(line)

    if not inserted:
        raise RuntimeError("Cannot find model.eval() or with torch.no_grad()")

    s = "\n".join(new_lines) + "\n"

# collect predictions
old = "        batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')"
new = """        preds = classification_logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        batch_f1_cls = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')"""

if old in s and "all_preds.extend" not in s:
    s = s.replace(old, new, 1)

# save CSV after final TEST log
marker = '    write_log(logger, f"Done! - TEST - Loss: {test_loss_cls:.4f} - Acc: {test_acc_cls:.4f} - F1: {test_f1_cls:.4f}")\n'

save_block = r'''
    save_pred_path = os.environ.get("SAVE_PRED")
    if save_pred_path:
        save_dir = os.path.dirname(save_pred_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        n = min(len(all_labels), len(all_preds))
        df = pd.DataFrame({
            "idx": list(range(n)),
            "gold": all_labels[:n],
            "pred": all_preds[:n],
        })
        df.to_csv(save_pred_path, index=False)
        write_log(logger, f"Saved predictions to {save_pred_path}")
'''

if "Saved predictions to {save_pred_path}" not in s:
    if marker not in s:
        raise RuntimeError("Cannot find final TEST log marker. Show: nl -ba task/classification/test.py | sed -n '1,150p'")
    s = s.replace(marker, marker + save_block, 1)

p.write_text(s)

print("Patched SAVE_PRED successfully.")
