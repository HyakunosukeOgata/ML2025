# Intelligent QA Pipeline Using Llama-3 and Google Search

## 📚 Features

### Question Extraction
Extract and refine the core intent from verbose user inputs.

### Keyword Extraction
Extract 2–5 essential keywords to assist in precise web searching.

### Smart Web Search (Google-Based)
Dynamically decide whether a web search is necessary based on question complexity.

### Context-Aware Answer Generation
Generate logic-driven and accurate answers by leveraging background knowledge and common sense.

### Batch Processing Support
Automatically process a large number of questions from `public.txt` and `private.txt`, and organize outputs by student ID.

### Robust Search Retry Mechanism
Implements an exponential backoff strategy with three attempts to maximize search reliability.

## ⚙️ Technologies

- Python 3.10+
- Llama-3 (Quantized model with `llama-cpp-python`)
- Asynchronous web scraping tools: `requests_html`, `asyncio`
- Natural language processing tools: `BeautifulSoup`, `charset_normalizer`
- Search integration: `googlesearch-python`

## 🚀 Usage

1. Place your Llama-3 model file (e.g., `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`) in the working directory.
2. Prepare `public.txt` and (optionally) `private.txt`, with one question per line.
3. Set your student ID in the `STUDENT_ID` variable inside the script.
4. Run:

   ```bash
   python your_script_name.py
