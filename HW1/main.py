import asyncio
from pathlib import Path
from typing import List

# ---------- Basic Utilities ----------
import urllib3
from llama_cpp import Llama
import torch

# ---------- 🔄 Search Related ----------
from bs4 import BeautifulSoup               # HTML parser
from charset_normalizer import detect        # Encoding detection
from googlesearch import search as _search   # Google search wrapper
from requests_html import AsyncHTMLSession   # Async web fetcher
# --------------------------------------

urllib3.disable_warnings()

# ===============================
# 1. Load Llama‑3 Model
# ===============================
llama3 = Llama(
    model_path="./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=32768,
)

def generate_response(_model: Llama, _messages: List[dict]) -> str:
    """Unified method to call the model"""
    response = _model.create_chat_completion(
        messages=_messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=512,
        temperature=0,
        repeat_penalty=2.0,
    )
    return response["choices"][0]["message"]["content"]

class LLMAgent:
    """Simple Multi-Agent Container"""
    def __init__(self, role_description: str, task_description: str):
        self.role_description = role_description
        self.task_description = task_description

    def inference(self, message: str) -> str:
        messages = [
            {"role": "system", "content": self.role_description},
            {"role": "user",  "content": f"Task Description: {self.task_description}\nQuestion: {message}"},
        ]
        return generate_response(llama3, messages)

# ===============================
# 2. Three Small Agents
# ===============================
question_extraction_agent = LLMAgent(
    role_description="You are a language processing expert specializing in extracting core questions from narratives.",
    task_description="Find out the real question intended, preserve the full meaning, remove unrelated content, and only tell me the core question in Traditional Chinese."
)

keyword_extraction_agent = LLMAgent(
    role_description="You are a keyword extraction expert good at extracting keywords that help understand intent.",
    task_description="Extract 2-5 keywords, and only tell me the keywords in Traditional Chinese."
)

qa_agent = LLMAgent(
    role_description="You are a knowledge-based QA system skilled in logical reasoning using background information.",
    task_description="Based on the provided context, answer the question specifically and logically in Traditional Chinese."
)

# ===============================
# 3. Search Utility Functions
# ===============================
async def _fetch(session: AsyncHTMLSession, url: str):
    """Async fetch page + MIME type checking"""
    try:
        head = await asyncio.wait_for(session.head(url, verify=False), timeout=10)
        if 'text/html' not in head.headers.get('Content-Type', ''):
            return None
        r = await asyncio.wait_for(session.get(url, verify=False), timeout=10)
        return r.text
    except:
        return None

async def get_htmls(urls: List[str]) -> List[str]:
    session = AsyncHTMLSession()
    tasks = (_fetch(session, u) for u in urls)
    return await asyncio.gather(*tasks)

async def search(query: str, n_results: int = 3) -> str:
    """Google precise search + extract plain text context"""
    quoted = f'"{query.strip()}"'
    urls = list(_search(quoted[:100], n_results * 2, lang="zh", unique=True))
    pages = await get_htmls(urls)
    pages = [p for p in pages if p]
    texts = [BeautifulSoup(p, 'html.parser').get_text(separator=' ', strip=True) for p in pages]

    clean = [''.join(t.split()) for t in texts if detect(t.encode()).get('encoding') == 'utf-8']
    return "\n\n".join(clean)[:5000] 

def should_use_search(question: str) -> bool:
    """Simple rule to determine whether to search"""
    return len(question) > 15 or any(
        kw in question for kw in ["what is", "who", "which"]
    )

# ===============================
# 4. Main Pipeline
# ===============================
import random, asyncio

async def pipeline(question: str) -> str:
    print(f"\n🟡 Original Question: {question}")

    # (1) Extract core question
    extracted = question_extraction_agent.inference(question)
    print(f"🔍 Extracted Question: {extracted}")

    # (2) Extract keywords
    keywords = keyword_extraction_agent.inference(extracted)
    print(f"🔑 Keywords: {keywords} + ")

    # (3) Decide if search is needed
    context = "No search needed, answer based on common sense."
    if should_use_search(extracted):
        # --- Add retry logic up to 3 times ---
        for attempt in range(3):
            try:
                context = await search(extracted)
                break
            except Exception as e:
                wait = 20 ** attempt + random.random()  # Exponential backoff + random delay
                print(f"⚠️ Attempt {attempt+1} Search Failed: {e}\n   Retry after {wait:.1f}s")
                await asyncio.sleep(wait)
        else:
            context = "Multiple search failures, answer based on common sense."

    print(f"🌐 Context Used (First 300 characters):\n{context[:300]}...\n")

    # (4) Let QA agent answer
    prompt = f"Background information:\n{context}\n\nPlease answer: {extracted}"
    answer = qa_agent.inference(prompt)
    print(f"✅ Final Answer: {answer}")
    return answer

# ===============================
# 5. Batch Processing Settings
# ===============================
STUDENT_ID = "your_student_id"  # <<< Change to your student ID
OUTPUT_DIR = Path(STUDENT_ID)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    # 5‑1 public.txt
    with open('./public.txt', 'r', encoding='utf-8') as f:
        questions = [line.strip().split(',')[0] for line in f.readlines()]
        for idx, q in enumerate(questions, 1):
            out = OUTPUT_DIR / f"{idx}.txt"
            if out.exists():
                continue
            answer = await pipeline(q)
            with open(out, 'w', encoding='utf-8') as f_out:
                f_out.write(answer.replace('\n', ' '))

    # # 5‑2 private.txt
    # with open('./private.txt', 'r', encoding='utf-8') as f:
    #     questions = [line.strip() for line in f.readlines()]
    #     for idx, q in enumerate(questions, 31):
    #         out = OUTPUT_DIR / f"{idx}.txt"
    #         if out.exists():
    #             continue
    #         answer = await pipeline(q)
    #         with open(out, 'w', encoding='utf-8') as f_out:
    #             f_out.write(answer.replace('\n', ' '))

    # # 5‑3 Merge
    # with open(f'./{STUDENT_ID}.txt', 'w', encoding='utf-8') as f_out:
    #     for i in range(1, 91):
    #         with open(f'{STUDENT_ID}/{i}.txt', 'r', encoding='utf-8') as f_in:
    #             f_out.write(f_in.readline().strip() + '\n')

# ===============================
# 6. Entry Point
# ===============================
if __name__ == '__main__':
    asyncio.run(main())  # Run public + private batch
    # asyncio.run()
