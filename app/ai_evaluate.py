"""Stage 2: AI evaluation via an OpenAI-compatible LLM API.

Reads every job that passed the deterministic filters (filters.py) but
hasn't been scored yet (dedup.get_unevaluated_candidates), sends it to the
configured LLM alongside your profile.yaml, and asks for a FUNCTIONAL FIT
judgment — not a title match. Results are stored in the ai_evaluations
table (see dedup.py) and written out to data/scored_candidates.csv, best
match first.

Talks to any OpenAI-compatible chat-completions endpoint (OpenAI itself,
Groq, a local router, ...) via LLM_BASE_URL/LLM_MODEL/LLM_API_KEY — see
.env.example. That's deliberate: which provider/model actually serves a
given run is meant to be swappable without touching this script.

This is the ONLY part of the pipeline that costs money — everything
upstream (fetch, filter, dedup) is free. That's the whole point of doing
filtering deterministically first: by the time a job reaches this script,
it's already passed title/location/stack screening, so the AI-scored
volume should be small.

Setup:
    pip install openai
    export LLM_API_KEY=...
    export LLM_MODEL=...                        # e.g. llama-3.3-70b-versatile
    export LLM_BASE_URL=...                      # e.g. https://api.groq.com/openai/v1 — omit for api.openai.com

Usage:
    python app/ai_evaluate.py              # evaluate everything unscored
    python app/ai_evaluate.py --limit 20   # cap this run (e.g. to control cost)
    python app/ai_evaluate.py --dry-run    # show what WOULD be sent, call nothing
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from app import dedup
from app import filters

load_dotenv()

MODEL = os.environ.get("LLM_MODEL")
OUTPUT_CSV = Path("data/scored_candidates.csv")

REQUIRED_EVAL_FIELDS = ["match_score", "recommendation", "genuine_gaps", "transferable_strengths", "risk_factors"]

# Field order matters here beyond documentation: tool-calling LLMs tend to
# emit JSON in roughly declaration order, and with max_tokens capped, a run
# of long free-text fields can eat the budget before later fields get
# written — which is exactly what caused a real KeyError on 'recommendation'
# in production (2026-08-11, see evaluate_one's retry logic below for the
# other half of the fix). Putting the two short/critical fields
# (match_score, recommendation) FIRST means they're very unlikely to be the
# ones lost to truncation even if a long-text field still gets cut off.
EVALUATION_SCHEMA = {
    "name": "submit_evaluation",
    "description": "Submit a structured fit evaluation for this job posting.",
    "input_schema": {
        "type": "object",
        "properties": {
            "match_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall functional fit score, 0-100.",
            },
            "recommendation": {
                "type": "string",
                "enum": ["apply", "consider", "skip"],
                "description": "apply = strong functional fit, worth the effort. consider = plausible but real gaps/risk. skip = not a genuine fit despite passing keyword filters.",
            },
            "genuine_gaps": {
                "type": "string",
                "description": "Real, specific gaps between the candidate's experience and this role's requirements. Be honest — don't invent gaps to seem balanced, and don't paper over real ones. Keep to 2-3 sentences.",
            },
            "transferable_strengths": {
                "type": "string",
                "description": "Which of the candidate's competencies/evidence genuinely transfer to this role, and why — cite specifics from their profile, not generic claims. Keep to 2-3 sentences.",
            },
            "risk_factors": {
                "type": "string",
                "description": "Non-skill risks: seniority mismatch, domain mismatch, likely comp mismatch, stack dealbreakers the deterministic filter might have missed, company-stage risk given the candidate's stated preferences, etc. Keep to 2-3 sentences.",
            },
        },
        "required": REQUIRED_EVAL_FIELDS,
    },
}


def _to_openai_tool(schema: dict) -> dict:
    """Reshapes an Anthropic-style tool schema (name/description/input_schema)
    into the OpenAI chat-completions tool format (name/description/parameters
    nested under function). The schema content itself — field names, types,
    descriptions — is untouched; this only changes the envelope."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        },
    }


class FatalLLMError(Exception):
    """Auth/credit/quota problems mean every subsequent call will fail the
    same way — raised to abort the whole run instead of burning through the
    remaining queue one failed job at a time."""


SYSTEM_PROMPT = """You are evaluating job postings for FUNCTIONAL FIT against a candidate's real \
experience — not title matching, not keyword matching. The candidate's profile is organized by \
competency (what they've actually done), not by job title, specifically so you judge whether their \
demonstrated capabilities transfer to this role's actual responsibilities.

Be honest and specific, not diplomatic. A generic "great candidate!" evaluation is useless — the \
candidate needs real signal on whether to spend an application on this. If the role is a stretch, \
say so and say why. If there's a real gap, name it precisely rather than softening it. Cite \
specific evidence from their profile when claiming a strength transfers; don't just assert \
seniority-level fit in the abstract."""


def load_profile(path: str = "profile.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# aggregator_clients.fetch_adzuna now tries to fetch the FULL job posting
# (see fetch_full_description there) instead of settling for Adzuna's short
# API snippet — that's the real fix for 2026-08-11's "Adzuna rows scoring
# low because the description was cut off mid-sentence" problem. This flag
# is the fallback for the cases that full-JD fetch still can't recover
# (site blocked scraping, dead redirect, genuinely short posting): tell the
# model explicitly rather than let a thin description read as "no
# responsibilities listed" and quietly tank match_score.
def build_user_prompt(profile: dict, job: dict) -> str:
    raw_description = job.get("description", "")
    description = filters.strip_html(raw_description).strip() or "(no JD text available)"
    partial_note = ""
    if raw_description and filters.looks_truncated(raw_description):
        partial_note = (
            "\nNOTE: this description looks like a short snippet, not the full posting — it may "
            "have been cut off mid-sentence. Do NOT lower match_score or invent genuine_gaps just "
            "because a responsibility/requirement isn't mentioned here; judge fit on title, "
            "company, location, and whatever specifics ARE present. If the snippet is too thin to "
            "say anything meaningful about stack or seniority, say so in genuine_gaps rather than "
            "guessing.\n"
        )
    return f"""CANDIDATE PROFILE:
{yaml.dump(profile, sort_keys=False, allow_unicode=True)}

---

JOB POSTING TO EVALUATE:
Company: {job['company']}
Title: {job['title']}
Location: {job['location']}
URL: {job['url']}
{partial_note}
Description:
{description}

---

Call submit_evaluation with your structured assessment."""


def _extract_tool_input(resp) -> dict | None:
    """Returns the parsed submit_evaluation arguments, or None if the model
    didn't call it, or raises json.JSONDecodeError if it called it with
    arguments that aren't valid JSON (OpenAI-format tool calls carry
    arguments as a raw string, unlike Anthropic's already-parsed dict)."""
    message = resp.choices[0].message
    for call in message.tool_calls or []:
        if call.function.name == "submit_evaluation":
            return json.loads(call.function.arguments)
    return None


def _missing_fields(evaluation: dict) -> list[str]:
    return [f for f in REQUIRED_EVAL_FIELDS if f not in evaluation]


def evaluate_one(client, profile: dict, job: dict, max_retries: int = 1) -> dict:
    """Calls the model and validates the tool response has every required
    field. A forced tool_choice on a smaller model can still emit a
    truncated/incomplete JSON object (this happened in production on
    2026-08-11 — see EVALUATION_SCHEMA's comment) if max_tokens is hit
    mid-generation; retry once with a bump to max_tokens before giving up,
    rather than crashing the whole run on one bad response.

    Auth/credit errors (openai.AuthenticationError, PermissionDeniedError,
    RateLimitError — the last covers both throttling and exhausted quota in
    the OpenAI/Groq error format) are not retried here: they propagate as
    FatalLLMError so the caller aborts the whole run instead of failing job
    by job."""
    import openai

    user_content = build_user_prompt(profile, job)
    max_tokens = 1536
    tool = _to_openai_tool(EVALUATION_SCHEMA)

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": "submit_evaluation"}},
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.RateLimitError) as e:
            raise FatalLLMError(f"Aborting: {type(e).__name__} from the LLM API — {e}") from e

        try:
            evaluation = _extract_tool_input(resp)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                max_tokens += 512
                continue
            raise RuntimeError(f"Model's response for {job['url']} wasn't valid JSON after "
                                f"{max_retries + 1} attempt(s): {e}") from e

        if evaluation is None:
            finish_reason = getattr(resp.choices[0], "finish_reason", "unknown")
            if attempt < max_retries:
                max_tokens += 512  # give the retry more room in case it was truncation
                continue
            raise RuntimeError(f"Model didn't call submit_evaluation for {job['url']} "
                                f"(finish_reason={finish_reason})")

        missing = _missing_fields(evaluation)
        if not missing:
            return evaluation
        if attempt < max_retries:
            max_tokens += 512
            continue
        raise RuntimeError(f"Model's response for {job['url']} is missing required field(s) "
                            f"{missing} after {max_retries + 1} attempt(s): {evaluation}")


def write_csv(conn, path: Path = OUTPUT_CSV) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(dedup.iter_scored_candidates(conn))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "match_score", "recommendation", "company", "title", "location",
            "transferable_strengths", "genuine_gaps", "risk_factors", "url", "posted_at",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max number of jobs to evaluate this run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated, call no API")
    args = parser.parse_args()

    profile = load_profile()

    with dedup.connect() as conn:
        queue = dedup.get_unevaluated_candidates(conn)
        if args.limit:
            queue = queue[: args.limit]

        print(f"{len(queue)} candidate(s) queued for AI evaluation.")
        if not queue:
            sys.exit(0)

        if args.dry_run:
            for job in queue:
                print(f"WOULD EVALUATE | {job['company']:20s} | {job['title']}")
            sys.exit(0)

        try:
            import openai
        except ImportError:
            sys.exit("Missing dependency: pip install openai")

        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            sys.exit("LLM_API_KEY env var not set.")
        if not MODEL:
            sys.exit("LLM_MODEL env var not set.")
        client = openai.OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL") or None)

        for i, job in enumerate(queue, 1):
            try:
                evaluation = evaluate_one(client, profile, job)
            except FatalLLMError as e:
                sys.exit(f"{e} — aborting run ({i - 1}/{len(queue)} evaluated so far this run).")
            except Exception as e:
                print(f"[WARN] {job['company']} — {job['title']}: evaluation failed — {e}", file=sys.stderr)
                continue
            dedup.save_evaluation(conn, job["url"], evaluation, MODEL)
            conn.commit()  # commit per-job so a crash mid-run doesn't lose completed evaluations
            print(f"[{i}/{len(queue)}] {evaluation['match_score']:3d} {evaluation['recommendation']:9s} | "
                  f"{job['company']:20s} | {job['title']}")

        total = write_csv(conn)
        print(f"\nWrote {total} scored candidates to {OUTPUT_CSV} (sorted by match_score desc).")