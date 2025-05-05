import subprocess
from pathlib import Path

# === CONFIGURE THESE ===
notebook_path = "Baseline_EfficientNet_multiclass_skincancer_torch.ipynb"  # Replace with your notebook
paired_script_ext = "py"             # Can be "py" or "md" (for markdown)

# === DO NOT MODIFY BELOW ===
notebook = Path(notebook_path)
script_path = notebook.with_suffix(f".{paired_script_ext}")

# Step 1: Pair the notebook with a script
print(f"Pairing {notebook} with {script_path}...")
subprocess.run(["jupytext", "--set-formats", f"ipynb,{paired_script_ext}", str(notebook)])

# Step 2: Sync notebook <-> script
print("Synchronizing...")
subprocess.run(["jupytext", "--sync", str(notebook)])

print(f"✅ Done. You can now edit {script_path} in Cursor. Changes will sync to {notebook.name}.")
