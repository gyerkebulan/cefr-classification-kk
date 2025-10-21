import argparse
from src.text.predict_text import predict_text_cefr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_kz", required=True, help="Kazakh text")
    ap.add_argument(
        "--text_ru",
        required=False,
        help="Russian translation (optional, skips automatic translation when provided)",
    )
    args = ap.parse_args()
    result = predict_text_cefr(args.text_kz, russian_text=args.text_ru)
    print("Translation:", result.translation)
    print("Text CEFR:", result.average_level)
    print("Distribution:", result.distribution)
    print("Phrase alignments (KZ → RU):")
    for phrase in result.phrase_alignments:
        print(f'  "{phrase.kazakh_phrase}" -> "{phrase.russian_token}"')

if __name__ == "__main__":
    main()
