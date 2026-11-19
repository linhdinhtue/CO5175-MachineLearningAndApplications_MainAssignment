"""Download Adult, record provenance/checksums and basic statistics."""
from src.data import ADULT_URL, FILES, RAW_DIR, describe, load_raw, sha256
from src.utils import env_info, output_dir, parse_config, save_json


def main() -> None:
    cfg = parse_config()
    out = output_dir("data_check")

    train, test = load_raw()
    meta = {
        "source_url": ADULT_URL,
        "files": {
            name: {"sha256": sha256(RAW_DIR / name), "bytes": (RAW_DIR / name).stat().st_size}
            for name in FILES
        },
        "duplicate_policy": cfg["duplicate_policy"],
        "train": describe(train),
        "test": describe(test),
        "dtypes": {c: str(t) for c, t in train.dtypes.items()},
    }
    save_json(meta, out / "data_meta.json")
    save_json(env_info(), out / "env.json")
    save_json(cfg, out / "config.json")

    train.describe(include="all").transpose().to_csv(out / "train_describe.csv")
    for col in train.select_dtypes(exclude="number").columns:
        train[col].value_counts(dropna=False).to_csv(out / f"counts_{col}.csv")

    print(f"train={meta['train']['n_rows']} test={meta['test']['n_rows']}")
    print("train labels:", meta["train"]["label_counts"])
    print("test labels:", meta["test"]["label_counts"])


if __name__ == "__main__":
    main()
