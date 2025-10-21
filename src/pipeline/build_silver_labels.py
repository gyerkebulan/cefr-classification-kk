from __future__ import annotations

from pathlib import Path

from cefr.alignment import EmbeddingAligner
from cefr.config import AlignmentConfig
from cefr.data import DEFAULT_RUS_CEFR
from cefr.data.silver import build_silver_labels

PARALLEL_CSV = Path("data/parallel/kazparc_kz_ru.csv")
RUS_CEFR = DEFAULT_RUS_CEFR
OUT_CSV = Path("data/labels/silver_word_labels.csv")


def main(
    parallel_csv: str | Path = PARALLEL_CSV,
    rus_cefr: str | Path = RUS_CEFR,
    out_csv: str | Path = OUT_CSV,
    aligner: EmbeddingAligner | None = None,
    layer: int = 8,
    thresh: float = 0.05,
    skip_non_informative: bool = True,
) -> Path:
    alignment_config = AlignmentConfig(layer=layer, threshold=thresh)
    return build_silver_labels(
        parallel_csv=parallel_csv,
        rus_cefr=rus_cefr,
        out_csv=out_csv,
        aligner=aligner,
        alignment_config=alignment_config,
        skip_non_informative=skip_non_informative,
    )


if __name__ == "__main__":
    main()
