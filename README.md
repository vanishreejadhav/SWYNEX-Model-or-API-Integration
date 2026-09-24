# SWYNEX-Model-or-API-Integration — AI Text Summarizer

A small command-line prototype that integrates a pretrained AI summarization
model into a working tool: paste in a long piece of text (or point it at a
file), and it returns a concise summary.

## How it works

The script uses the [Hugging Face Transformers](https://huggingface.co/docs/transformers)
library and the `facebook/bart-large-cnn` model — a model fine-tuned
specifically for text summarization. The model runs **locally on your
machine** the first time you use it (Transformers downloads and caches the
weights automatically), so **no API key or account is required**.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Summarize raw text directly:

```bash
python summarizer.py --text "Your long paragraph of text goes here..."
```

Summarize the contents of a file:

```bash
python summarizer.py --file examples/example_input.txt
```

Optional flags:

```bash
python summarizer.py --file examples/example_input.txt --max-length 80 --min-length 20
```

| Flag           | Description                                    | Default              |
|----------------|-------------------------------------------------|----------------------|
| `--text`       | Raw text to summarize                          | —                    |
| `--file`       | Path to a `.txt` file to summarize             | —                    |
| `--max-length` | Maximum length of the generated summary        | 130                  |
| `--min-length` | Minimum length of the generated summary        | 30                   |
| `--model`      | Hugging Face model to use                      | facebook/bart-large-cnn |

## Example

Input: [`examples/example_input.txt`](examples/example_input.txt) — a short
paragraph about AI's impact across industries.

Output: see [`examples/example_output.txt`](examples/example_output.txt) for
the expected result. On first run, Transformers will download the model
(~1.6 GB) and cache it, so the first summary takes 10-30 seconds longer than
subsequent ones.

## Project structure

```
SWYNEX-Model-or-API-Integration/
├── summarizer.py              # Main CLI script
├── requirements.txt           # Dependencies
├── examples/
│   ├── example_input.txt      # Sample input text
│   └── example_output.txt     # Sample expected output
├── .gitignore
└── README.md
```

## Notes

- No secret keys or credentials are used or required by this project.
- The model download requires an internet connection on first run only;
  after that it's cached locally and works offline.
