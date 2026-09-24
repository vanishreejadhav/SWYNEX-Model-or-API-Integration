"""
AI Text Summarizer (CLI)
-------------------------
Integrates a pretrained AI summarization model (facebook/bart-large-cnn via
Hugging Face Transformers) into a small command-line prototype.

Usage:
    python summarizer.py --text "Some long text to summarize..."
    python summarizer.py --file example_input.txt
    python summarizer.py --file example_input.txt --max-length 80 --min-length 20

No API key is required - the model runs locally on your machine.
"""

import argparse
import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_MODEL = "facebook/bart-large-cnn"


def load_summarizer(model_name: str = DEFAULT_MODEL):
    """Load and return the tokenizer and model needed for summarization.

    Loaded directly via AutoTokenizer / AutoModelForSeq2SeqLM (rather than the
    pipeline() shortcut) so this works consistently across transformers versions.
    """
    print(f"Loading model '{model_name}' (this may take a moment the first time)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def summarize_text(summarizer, text, max_length=130, min_length=30):
    """Generate a summary for the given text using the loaded tokenizer/model."""
    if not text or not text.strip():
        raise ValueError("Input text is empty. Please provide text to summarize.")

    tokenizer, model = summarizer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        do_sample=False,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def get_input_text(args):
    """Resolve the input text from either --text or --file."""
    if args.text:
        return args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("You must provide either --text or --file as input.")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize text using a pretrained AI summarization model."
    )
    parser.add_argument("--text", type=str, help="Raw text to summarize.")
    parser.add_argument("--file", type=str, help="Path to a .txt file containing text to summarize.")
    parser.add_argument("--max-length", type=int, default=130, help="Maximum length of the summary (default: 130).")
    parser.add_argument("--min-length", type=int, default=30, help="Minimum length of the summary (default: 30).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model to use.")

    args = parser.parse_args()

    try:
        input_text = get_input_text(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    summarizer = load_summarizer(args.model)
    summary = summarize_text(summarizer, input_text, args.max_length, args.min_length)

    print("\n----- ORIGINAL TEXT (first 300 chars) -----")
    print(input_text.strip()[:300] + ("..." if len(input_text.strip()) > 300 else ""))
    print("\n----- SUMMARY -----")
    print(summary)


if __name__ == "__main__":
    main()
