# Sentiment-Analysis-Using-Multi-Agent-Framework
**Project overview:**
**This repository** contains an agentic-AI equivalent implementation of the ADRSA multi-agent sentiment analysis framework. It implements a pipeline that splits and preprocesses review text, detects aspects, identifies sentiment-bearing words, detects negation and its scope, and generates final aspect–polarity label pairs.

**Academic reference:**
- Reka K, Raja G, Malarchelvi PD SK. AI and Deep Reinforcement Learning for Sentiment Analysis of Customer Reviews. Web Intelligence. 2025;23(2):213-221. doi:10.1177/24056456241304010

This implementation follows the approach described in the paper (the proposed multi-agent ADRSA framework). The pipeline mirrors the paper's stages:
- Preprocessing: extract meaningful sentences and remove sentences without sentiment words.
- Aspect detection: identify product/service aspects (uses n-gram analysis and a POS fallback).
- Sentiment word detection: detect and normalize sentiment-bearing words and intensities.
- Negation detection: find negation terms and determine whether polarity should be flipped.
- Label pair generation: combine the above to produce (aspect, polarity, confidence) pairs.

**Contents**
- `adrsa_framework.py`: Main implementation of the ADRSA pipeline and agent wrappers.
- `download_nltk.py`: helper to download required NLTK models.
- `requirements.txt`: Python dependencies.

**Quickstart — prerequisites**
- **Python**: 3.10+ recommended.
- Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Download required NLTK data (the repo includes `download_nltk.py`):

```bash
python download_nltk.py
```

**Environment variables**
- `GROQ_API_KEY` (required for LLM calls): API key for the Groq client used in the implementation.
- `LOG_LEVEL` (optional): set to `DEBUG` for verbose tracing or `INFO` for concise progress logs. Default is `INFO`.

Example:

```bash
export GROQ_API_KEY="your-key-here"
export LOG_LEVEL=DEBUG
python3 adrsa_framework.py
```

**How it works — implementation notes**
- The pipeline reads a full review, calls the preprocessing agent to split and filter sentences, then processes each sentence through the four analysis agents, and finally generates label pairs.
- Each LLM call is wrapped in a try/except. If the LLM fails or returns noisy text, the system attempts to extract a JSON substring using a robust `_extract_json_from_text` helper. If JSON extraction or parsing fails, safe fallbacks are used:
	- Preprocessing: regex-based sentence splitting.
	- Aspect detection: NLTK POS-tagging fallback to extract noun phrases.
	- Sentiment/negation: return safe default structures indicating no sentiment/negation.
	- Label pair generation: normalize polarity values and confidence; skip malformed entries.

**Design choices for robustness**
- JSON extraction helper: scans LLM output and attempts to parse the first valid JSON object/array. This allows tolerant parsing when LLMs add explanations or text around JSON.
- Conservative fallbacks: ensure the pipeline continues even when an agent fails; per-sentence exceptions are caught so one problematic sentence does not stop the whole review analysis.
- Normalization: polarity strings and confidence values are normalized, with unknown/invalid entries mapped to safe defaults.
- Logging: `adrsa_framework.py` contains structured logging controlled by `LOG_LEVEL`. Set to `DEBUG` to see raw LLM response lengths, extracted JSON blocks, and parsing decisions.

**Usage example**
1. Start with environment variables and NLTK download (see above).
2. Run the CLI:

```bash
python3 adrsa_framework.py
```

You will be prompted to paste a multi-line review (press Enter twice to finish input). The script prints the final aspect–polarity pairs with confidence scores.

**Developer notes / testing**
- The repo includes a helper `download_nltk.py` for NLTK corpora. Ensure you run it in the same environment as the code.
- To test robustness of parsing logic, consider adding unit tests that feed example noisy LLM outputs to `_extract_json_from_text` and the parse functions.

**Limitations & performance**
- The current implementation uses LLM calls synchronously via `self.client.chat.completions.create`. For large batches or production use, consider batching, concurrency limits, and rate-limit handling.
- Building good aspect and sentiment vocabularies (n-gram models) requires many iterations and data: the original paper notes thousands of iterations to converge.

**Cite**
If you use this code in your research, please cite the originating paper:

Reka K, Raja G, Malarchelvi PD SK. AI and Deep Reinforcement Learning for Sentiment Analysis of Customer Reviews. Web Intelligence. 2025;23(2):213-221. doi:10.1177/24056456241304010