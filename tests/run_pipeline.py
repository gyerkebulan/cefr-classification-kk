import argparse

from cefr import TextPipeline, load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_kz", required=True, help="Kazakh text")
    ap.add_argument(
        "--text_ru",
        required=False,
        help="Russian translation (optional, skips automatic translation when provided)",
    )
    args = ap.parse_args()
    config = load_config()
    pipeline = TextPipeline(config=config.pipeline)
    prediction = pipeline.predict(args.text_kz, russian_text=args.text_ru)
    print("Translation:", prediction.translation)
    print("Text CEFR:", prediction.average_level)
    print("Distribution:", dict(prediction.distribution))
    print("Phrase alignments (KZ → RU):")
    for phrase in prediction.phrase_alignments:
        print(f'  "{phrase.kazakh_phrase}" -> "{phrase.russian_token}"')

if __name__ == "__main__":
    main()
