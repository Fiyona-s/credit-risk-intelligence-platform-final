"""Natural language -> SQL via Groq (primary, free-tier) or Gemini (fallback, free-tier).

No paid LLM API is used anywhere in this module. Groq is tried first because
its free tier is fast and has generous rate limits for an interactive chat UI;
Gemini's free tier is the fallback if GROQ_API_KEY is unset or the call fails.
"""
from typing import List, Optional, Tuple

from src.talk_to_data.prompt_templates import (
    ANSWER_SUMMARY_PROMPT_TEMPLATE,
    RETRY_PROMPT_TEMPLATE,
    SCHEMA_DESCRIPTION,
    SYSTEM_PROMPT_V1,
)
from src.talk_to_data.query_runner import SQLValidationError, run_query
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def _call_groq(system_prompt: str, user_message: str) -> str:
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(system_prompt: str, user_message: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model, system_instruction=system_prompt)
    response = model.generate_content(user_message)
    return response.text.strip()


def call_llm(system_prompt: str, user_message: str) -> str:
    """Try Groq first, fall back to Gemini. Raises RuntimeError if neither is configured/working."""
    if settings.has_groq:
        try:
            return _call_groq(system_prompt, user_message)
        except Exception as e:
            log.warning(f"Groq call failed ({e}), falling back to Gemini")
    if settings.has_gemini:
        try:
            return _call_gemini(system_prompt, user_message)
        except Exception as e:
            log.error(f"Gemini call also failed: {e}")
            raise RuntimeError("Both Groq and Gemini calls failed") from e
    raise RuntimeError(
        "No LLM provider configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env"
    )


def _strip_sql_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text.strip()


def ask(question: str, conversation_history: Optional[List[Tuple[str, str]]] = None) -> dict:
    """Answer a natural-language question by generating SQL, validating it, running it, and summarizing.

    conversation_history: list of (question, answer_summary) tuples for the last few turns,
    so follow-ups like "and for females only?" resolve correctly.
    """
    history_text = ""
    if conversation_history:
        recent = conversation_history[-3:]
        history_text = "\n".join(f"Previous Q: {q}\nPrevious A: {a}" for q, a in recent)
        history_text = f"\nConversation so far:\n{history_text}\n"

    user_message = f"{history_text}\nQuestion: {question}\nSQL:"

    sql_raw = call_llm(SYSTEM_PROMPT_V1, user_message)
    sql = _strip_sql_fences(sql_raw)

    try:
        result_df, clean_sql = run_query(sql)
    except SQLValidationError as e:
        log.warning(f"Invalid SQL on first attempt: {e}. Retrying once.")
        retry_message = RETRY_PROMPT_TEMPLATE.format(error=str(e), previous_sql=sql, schema=SCHEMA_DESCRIPTION)
        sql_raw_2 = call_llm(SYSTEM_PROMPT_V1, retry_message)
        sql_2 = _strip_sql_fences(sql_raw_2)
        try:
            result_df, clean_sql = run_query(sql_2)
            sql = sql_2
        except SQLValidationError as e2:
            log.error(f"Invalid SQL on retry too: {e2}")
            return {
                "success": False,
                "message": (
                    "I couldn't turn that into a safe query against the applicant data. "
                    "Try rephrasing, e.g. 'average income of defaulters' or 'default rate by education level'."
                ),
                "sql": None,
                "result": None,
            }

    preview = result_df.head(5).to_dict(orient="records")
    summary_prompt = ANSWER_SUMMARY_PROMPT_TEMPLATE.format(
        question=question, sql=clean_sql, result_preview=preview
    )
    try:
        summary = call_llm("You are a concise, accurate credit risk analyst.", summary_prompt)
    except RuntimeError:
        summary = "Here are the results (LLM summary unavailable)."

    return {
        "success": True,
        "message": summary,
        "sql": clean_sql,
        "result": result_df,
    }


if __name__ == "__main__":
    result = ask("What is the average income of defaulters?")
    print(result["sql"])
    print(result["result"])
    print(result["message"])
