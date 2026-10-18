"""Download the external data used by Nesting Ralphs into data/ (not committed).

- Pi Agent sessions: championswimmer/pi-coding-sessions (MIT), HF `format:agent-traces`
- PleIAs/SYNTH shard synth_001.parquet, rows with exercise == 'math exercise' (CC-BY-SA 4.0)

Usage: uv run python scripts/fetch_data.py
"""

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
PI_REPO = "championswimmer/pi-coding-sessions"


def fetch_pi():
    out = ROOT / "data/pi"
    out.mkdir(parents=True, exist_ok=True)
    for f in HfApi().list_repo_files(PI_REPO, repo_type="dataset"):
        if f.endswith(".jsonl"):
            p = hf_hub_download(PI_REPO, f, repo_type="dataset")
            (out / Path(f).name).write_bytes(Path(p).read_bytes())
    print("pi sessions:", len(list(out.glob("*.jsonl"))))


def fetch_synth():
    out = ROOT / "data/synth/math_exercise_synth_001.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = pq.ParquetFile(HfFileSystem().open("datasets/PleIAs/SYNTH/synth_001.parquet", block_size=16 << 20))
    t = f.read(columns=["synth_id", "language", "exercise", "model", "query", "synthetic_reasoning",
                        "synthetic_answer", "seed_license"])
    t = t.filter(pc.equal(t["exercise"], "math exercise"))
    with open(out, "w") as o:
        for r in t.to_pylist():
            o.write(json.dumps(r) + "\n")
    print("synth math rows:", t.num_rows)


if __name__ == "__main__":
    fetch_pi()
    fetch_synth()
