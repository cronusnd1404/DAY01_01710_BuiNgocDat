"""
K4 — Ngày 1: Khám Phá LLM API (14h00–18h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
import sys
from typing import Any, Callable

from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-5.5": {"input": 0.0025, "output": 0.010},
    "gpt-5.6": {"input": 0.0015, "output": 0.006},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
}

# Luồng chính: OpenAI (mặc định gpt-5.5 và gpt-5.6)
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-5.5")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-5.5")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 15h00–15h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API, trả về nội dung phản hồi + độ trễ.

    Args:
        prompt:      Tin nhắn của người dùng.
        model:       Model OpenAI sử dụng (mặc định: gpt-4o).
        temperature: Độ ngẫu nhiên khi lấy mẫu (0.0 – 2.0).
        top_p:       Ngưỡng nucleus sampling.
        max_tokens:  Số token tối đa được sinh ra.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        from openai import OpenAI            # import BÊN TRONG hàm
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # đo thời gian bằng time.time() trước và sau lời gọi API
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    if latency <= 0:
        latency = 0.0001

    response_text = response.choices[0].message.content
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        Tái sử dụng call_openai() với model=OPENAI_MINI_MODEL — 1 dòng code.
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.

    Returns:
        Dict với các key:
            - "gpt4o_answer":      str
            - "mini_answer":       str
            - "gpt4o_time":       float
            - "mini_time":        float
            - "gpt4o_cost": float  (USD ước tính cho phản hồi)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(
            OPENAI_MODEL, PRICING_PER_1K_TOKENS["gpt-4o"]
        )
        cost = (len(response.split()) / 0.75) / 1000 * pricing["output"]
        (0.75 từ ≈ 1 token — ước lượng thô; Part 2 sẽ tính chính xác hơn.
         Dùng .get để lấy đúng giá model đang chạy — gpt-4o, gemini...;
         model không có trong bảng thì lấy giá gpt-4o làm tham chiếu)
    """
    gpt4o_answer, gpt4o_time = call_openai(prompt)
    mini_answer, mini_time = call_openai_mini(prompt)

    pricing = PRICING_PER_1K_TOKENS.get(
        OPENAI_MODEL, PRICING_PER_1K_TOKENS["gpt-4o"]
    )
    tokens_est = len(gpt4o_answer.split()) / 0.75
    gpt4o_cost = (tokens_est / 1000.0) * pricing["output"]

    return {
        "gpt4o_answer": gpt4o_answer,
        "mini_answer": mini_answer,
        "gpt4o_time": gpt4o_time,
        "mini_time": mini_time,
        "gpt4o_cost": gpt4o_cost,
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 15h40–16h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    if latency <= 0:
        latency = 0.0001

    response_text = response.choices[0].message.content
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    prompt_tokens = count_tokens(prompt, model=model)
    completion_tokens = count_tokens(response, model=model)

    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    prompt_cost = (prompt_tokens / 1000.0) * pricing["input"]
    completion_cost = (completion_tokens / 1000.0) * pricing["output"]
    total_cost = prompt_cost + completion_cost

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 16h30–17h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    history = []

    while True:
        try:
            user_msg = input("You: ")
        except (EOFError, StopIteration):
            break

        if user_msg.strip().lower() in ("quit", "exit", "bye"):
            break

        history.append({"role": "user", "content": user_msg})
        messages = history[-8:]

        try:
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
            reply = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end="", flush=True)
                    reply += delta
            print()
            history.append({"role": "assistant", "content": reply})
            history = history[-8:]
        except Exception as e:
            print(f"\nLỗi khi gọi API: {e}")
            break


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
    if last_exception:
        raise last_exception


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 17h10–17h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    from openai import OpenAI

    if get_input is None:
        get_input = input

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    history = []
    turns = 0
    tokens_used = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and turns >= max_turns:
            break

        try:
            user_msg = get_input()
        except (EOFError, StopIteration):
            break

        if user_msg.strip().lower() in ("quit", "exit", "bye"):
            break

        messages = [{"role": "system", "content": persona}] + history + [{"role": "user", "content": user_msg}]

        def do_call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )

        try:
            stream = retry_with_backoff(do_call)
            reply = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end="", flush=True)
                    reply += delta
            print()
        except Exception as e:
            print(f"\nLỗi khi gọi API: {e}")
            break

        turns += 1
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-8:]

        cost_info = estimate_cost(user_msg, reply, model=OPENAI_MODEL)
        tokens_used += cost_info["prompt_tokens"] + cost_info["completion_tokens"]
        total_cost += cost_info["total_cost"]

    return {
        "turns": turns,
        "tokens_used": tokens_used,
        "total_cost": total_cost,
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.
    """
    results = []
    for p in prompts:
        res = compare_models(p)
        res["prompt"] = p
        results.append(res)
    return results


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.
    """
    lines = [
        "| Prompt | Model 5.5 Response | Model 5.6 Response | 5.5 Latency | 5.6 Latency |",
        "|---|---|---|---|---|"
    ]
    for r in results:
        p_short = (r['prompt'][:37] + '...') if len(r['prompt']) > 40 else r['prompt']
        m55 = (r['gpt4o_answer'][:37] + '...') if len(r['gpt4o_answer']) > 40 else r['gpt4o_answer']
        m56 = (r['mini_answer'][:37] + '...') if len(r['mini_answer']) > 40 else r['mini_answer']
        lines.append(f"| {p_short} | {m55} | {m56} | {r['gpt4o_time']:.2f}s | {r['mini_time']:.2f}s |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
