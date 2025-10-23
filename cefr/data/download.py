import csv
from pathlib import Path

from datasets import load_dataset


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_kz_ru(
    split="train",
    out_dir="data/parallel",
    out_name="kazparc_kz_ru.csv",
):
    out_dir_path = Path(out_dir)
    ensure_dir(out_dir_path)
    ds = load_dataset("issai/kazparc", split=split)
    path = out_dir_path / out_name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kaz", "rus"])
        for ex in ds:
            kk = ex.get("kk")
            ru = ex.get("ru")
            if kk and ru:
                kk = kk.replace("\n", " ").strip()
                ru = ru.replace("\n", " ").strip()
                writer.writerow([kk, ru])
    with path.open(encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle) - 1
    print(f"Saved: {path} rows: {row_count}")
    return path


__all__ = ["save_kz_ru", "ensure_dir"]


def main():
    save_kz_ru()


if __name__ == "__main__":
    main()
