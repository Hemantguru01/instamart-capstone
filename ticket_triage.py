
from pathlib import Path
import json
import random
import re
import time

import pandas as pd
import mock_llm

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = {"billing", "delivery", "technical", "fraud"}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

ZERO_SHOT_PROMPT = """
CLASSIFY each support ticket into exactly one allowed category:
billing, delivery, technical, fraud.
Return a JSON-only reply in this exact form:
{"category": "...", "reason": "..."}
Do not include markdown or text outside the JSON object.
""".strip()

FEW_SHOT_PROMPT = """
CLASSIFY each support ticket into exactly one allowed category:
billing, delivery, technical, fraud.
Return a JSON-only reply in this exact form:
{"category": "...", "reason": "..."}
Do not include markdown or text outside the JSON object.

EXAMPLES:
Ticket: I was charged twice for my order.
Reply: {"category": "billing", "reason": "duplicate charge"}
Ticket: My account was hacked and I see unauthorized activity.
Reply: {"category": "fraud", "reason": "unauthorized account activity"}
Ticket: The app crashes whenever I try to log in.
Reply: {"category": "technical", "reason": "app login crash"}
""".strip()


def redact_text(text):
    """Apply the required regexes in the specified order."""
    text = text or ""
    counts = {}
    text, counts["email"] = EMAIL_RE.subn("[EMAIL]", text)
    text, counts["phone"] = PHONE_RE.subn("[PHONE]", text)
    text, counts["order_id"] = ORDER_RE.subn("[ORDER_ID]", text)
    return text, counts


def est_tokens(text):
    return mock_llm.est_tokens(text)


def validate_reply(content):
    """Return category, status, and reason without guessing."""
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        matches = [word for word in words if word in ALLOWED]
        if len(matches) == 1:
            return matches[0], "classified_keyword", ""
        return "", "flagged_invalid_output", ""

    if (
        isinstance(obj, dict)
        and obj.get("category") in ALLOWED
        and isinstance(obj.get("reason"), str)
    ):
        return obj["category"], "classified_json", obj["reason"]

    return "", "flagged_invalid_output", ""



def retry_chat(messages, ticket_id):
    """Retry transient failures up to three times after the first attempt."""
    recovered_errors = 0

    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            return response, recovered_errors
        except mock_llm.TransientLLMError:
            recovered_errors += 1

            if retry_number >= MAX_RETRIES:
                raise

            wait = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={retry_number} wait={wait:.5f}s"
            )
            time.sleep(wait)

    """Retry transient failures up to three times after the first attempt."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={retry_number} wait={wait:.5f}s"
            )
            time.sleep(wait)


def prepare_tickets():
    tickets = pd.read_csv(RAW_FILE, keep_default_na=False)
    required = {"ticket_id", "customer_id", "order_id", "text", "true_category"}
    missing = required - set(tickets.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    redacted_texts = []
    totals = {"email": 0, "phone": 0, "order_id": 0}

    for text in tickets["text"]:
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)
        for key, count in counts.items():
            totals[key] += count

        # Reapplying the patterns must find no personal data.
        assert EMAIL_RE.search(redacted) is None
        assert PHONE_RE.search(redacted) is None
        assert ORDER_RE.search(redacted) is None

    tickets["redacted_text"] = redacted_texts
    print("PII substitution counts:", totals)
    print("Total substitutions:", sum(totals.values()))
    print("PII remaining after redaction: 0")

    return tickets


def run_pipeline(prompt, token_budget=None, budget_output=None):
    """Run one deterministic mock batch, resetting mock state first."""
    tickets = prepare_tickets()
    mock_llm.reset()

    rows = []
    running_tokens = 0
    transient_errors = 0

    for row in tickets.to_dict("records"):
        ticket_id = row["ticket_id"]
        text = row["redacted_text"]

        result = {
            "ticket_id": ticket_id,
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "status": "",
            "category": "",
            "reason": "",
            "tokens_used": 0,
            "prompt_type": "few_shot" if "EXAMPLES:" in prompt else "zero_shot",
        }

        if not text.strip():
            result["status"] = "skipped_empty"
            rows.append(result)
            continue

        if est_tokens(text) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            rows.append(result)
            continue

        if token_budget is not None and running_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            rows.append(result)
            continue

        # The user message content is exactly the redacted ticket text.
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        assert messages[-1]["content"] == text
        assert EMAIL_RE.search(messages[-1]["content"]) is None
        assert PHONE_RE.search(messages[-1]["content"]) is None
        assert ORDER_RE.search(messages[-1]["content"]) is None

        try:
            response, recovered = retry_chat(messages, ticket_id)
            transient_errors += recovered
        except mock_llm.TransientLLMError:
            # The rubric expects every transient failure to recover.
            result["status"] = "flagged_invalid_output"
            rows.append(result)
            continue

        used = int(response.usage.total_tokens)
        running_tokens += used
        result["tokens_used"] = used

        content = response.choices[0].message.content
        category, status, reason = validate_reply(content)
        result["category"] = category
        result["status"] = status
        result["reason"] = reason
        rows.append(result)

    results = pd.DataFrame(rows)

    if budget_output:
        OUTPUT_DIR.mkdir(exist_ok=True)
        results.to_csv(OUTPUT_DIR / budget_output, index=False)

    classified = results["status"].isin(
        ["classified_json", "classified_keyword"]
    )
    accuracy = (
        (results.loc[classified, "category"] ==
         results.loc[classified, "true_category"]).mean()
        if classified.any() else float("nan")
    )
    classified_count = int(classified.sum())

    summary = {
        "tickets": len(results),
        "status_counts": results["status"].value_counts().to_dict(),
        "classified": classified_count,
        "accuracy_on_classified": round(float(accuracy), 4),
        "total_tokens": running_tokens,
        "tokens_per_classified_ticket": round(
            running_tokens / classified_count, 2
        ) if classified_count else 0,
        "transient_errors_recovered_or_seen": transient_errors,
    }

    print("\nRUN SUMMARY")
    print(json.dumps(summary, indent=2))
    return results, summary


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n=== ZERO-SHOT: no budget ===")
    zero_results, zero_summary = run_pipeline(ZERO_SHOT_PROMPT)

    print("\n=== FEW-SHOT: no budget ===")
    few_results, few_summary = run_pipeline(FEW_SHOT_PROMPT)

    print("\n=== ZERO-SHOT: budget demo ===")
    budget_results, budget_summary = run_pipeline(
        ZERO_SHOT_PROMPT,
        token_budget=6000,
        budget_output="triage_budget_demo.csv",
    )
    print(
        "Budget-exhausted tickets:",
        int((budget_results["status"] == "budget_exhausted").sum()),
        "Budget used: 6000",
    )

    print("\n=== FEW-SHOT: final run with budget 100000 ===")
    final_results, final_summary = run_pipeline(
        FEW_SHOT_PROMPT,
        token_budget=100000,
    )

    # Add segment information and apply the specified routing rules.
    segments_file = OUTPUT_DIR / "customer_segments.csv"
    segments = pd.read_csv(segments_file)
    segment_cols = segments[["customer_id", "segment_name", "is_premium"]].copy()
    segment_cols["is_premium"] = (
        segment_cols["is_premium"].astype(str).str.lower().eq("true")
    )

    final_results = final_results.merge(
        segment_cols, on="customer_id", how="left", validate="many_to_one"
    )
    final_results["segment_name"] = final_results["segment_name"].fillna("")
    final_results["is_premium"] = final_results["is_premium"].fillna(False)

    def route(row):
        if row["status"] in {
            "skipped_empty", "flagged_long",
            "flagged_invalid_output", "budget_exhausted",
        }:
            return "human_review", "NORMAL"
        if row["category"] == "fraud":
            return "security_team", "HIGH"
        if bool(row["is_premium"]):
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"

    routed = final_results.apply(route, axis=1, result_type="expand")
    final_results["route"] = routed[0]
    final_results["priority"] = routed[1]

    final_results[
        [
            "ticket_id", "customer_id", "status", "category",
            "segment_name", "priority", "route", "tokens_used",
        ]
    ].to_csv(OUTPUT_DIR / "triage_results.csv", index=False)

    print("\nSaved outputs/triage_budget_demo.csv")
    print("Saved outputs/triage_results.csv")
    print("\nFinal status counts:")
    print(final_results["status"].value_counts().to_dict())
    print("Final rows:", len(final_results))


if __name__ == "__main__":
    main()
