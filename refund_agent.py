
import csv
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mock_llm
from policy_rag import PolicyRAG


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"
STATE_DIR = ROOT / "state"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

rag = PolicyRAG(overlap=10)


# ---------------------------------------------------------------------
# Task 5: tools
# ---------------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order and return its status, amount, and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant policy chunks for a policy query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the late-delivery refund amount for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund in the ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human approval in the ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]


def get_order(order_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": row["order_amount"],
        "delivery_minutes": row["delivery_minutes"],
    }


def search_policy(query):
    return {"results": rag.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        return {"refund_inr": 0.0}

    return {"refund_inr": round(float(order_amount), 2)}


def _append_ledger(tool_name, order_id, refund_inr):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }

    # One JSON object per line.
    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {"refund_inr": round(float(refund_inr), 2)}


def issue_refund(order_id, refund_inr):
    return _append_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _append_ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ---------------------------------------------------------------------
# Helpers for mock_llm's tool-call format
# ---------------------------------------------------------------------

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _tool_call_data(tool_call):
    """Support the supplied mock's object format and dictionary format."""
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
    else:
        name = tool_call.name
        arguments = getattr(tool_call, "arguments", {})
        call_id = getattr(tool_call, "id", f"call_{uuid.uuid4().hex[:8]}")

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    return call_id, name, arguments


def _write_checkpoint(run_id, scenario_id, messages, tokens_used,
                      steps, tools_executed, llm_calls):
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": len(steps),
        "steps": steps,
        "tools_executed": tools_executed,
        "llm_calls": llm_calls,
    }

    path = STATE_DIR / f"{run_id}.json"
    path.write_text(
    json.dumps(checkpoint, indent=2, default=_json_default),
    encoding="utf-8",
    )


def _result(status, final_answer, steps, llm_calls, tokens_used, run_id):
    return {
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "run_id": run_id,
    }


def _status_from_answer(answer):
    if ":" in answer:
        return answer.split(":", 1)[0].strip()
    return "UNKNOWN"


# ---------------------------------------------------------------------
# Tasks 6-8: ReAct loop, safety limits, checkpointing, resumption
# ---------------------------------------------------------------------

def _continue_agent(
    run_id,
    checkpoint,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None,
):
    scenario_id = checkpoint["scenario_id"]
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    steps = checkpoint["steps"]
    tools_executed = checkpoint["tools_executed"]
    llm_calls = checkpoint["llm_calls"]

    started = time.monotonic()

    # Treat max_iterations as the maximum number of model calls for
    # this complete run, including calls made before a resume.
    while llm_calls < max_iterations:
        elapsed = time.monotonic() - started
        if elapsed >= time_limit_seconds:
            return _result(
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: handing this request to a human agent.",
                steps, llm_calls, tokens_used, run_id,
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        usage = getattr(response, "usage", None)
        tokens_used += int(getattr(usage, "total_tokens", 0) or 0)

        # The token limit is checked before any requested tool executes.
        if tokens_used > token_limit:
            return _result(
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: token budget exceeded; human review required.",
                steps, llm_calls, tokens_used, run_id,
            )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            final_answer = message.content or ""
            return _result(
                _status_from_answer(final_answer),
                final_answer,
                steps, llm_calls, tokens_used, run_id,
            )

        assistant_calls = []
        parsed_calls = []

        for tool_call in tool_calls:
            call_id, name, arguments = _tool_call_data(tool_call)

            if name not in TOOL_MAP:
                raise ValueError(f"Unknown tool requested by model: {name}")

            assistant_calls.append({
                "id": call_id,
                "name": name,
                "arguments": arguments,
            })
            parsed_calls.append((call_id, name, arguments))

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls,
        })

        for call_id, name, arguments in parsed_calls:
            observation = TOOL_MAP[name](**arguments)
            observation_json = json.dumps(
                observation, ensure_ascii=False, default=_json_default
            )

            step = {
                "tool_name": name,
                "arguments": arguments,
                "observation": observation,
            }
            steps.append(step)
            tools_executed.append(name)

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": observation_json,
            })

            # Save after every observation, before simulating a crash.
            _write_checkpoint(
                run_id, scenario_id, messages, tokens_used,
                steps, tools_executed, llm_calls,
            )

            if (
                crash_after_tools is not None
                and len(tools_executed) == crash_after_tools
            ):
                raise RuntimeError(
                    f"Simulated crash after tool {len(tools_executed)} "
                    f"({name}); checkpoint has been saved."
                )

    return _result(
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: step limit reached; human handoff required.",
        steps, llm_calls, tokens_used, run_id,
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
):
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:10]}"

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT: You are an order refund agent. Use the supplied "
                    "tools to look up orders, check policy, calculate refunds, "
                    "and issue or escalate eligible requests. Never issue a "
                    "refund for a non-completed order. Only completed orders "
                    "delivered in more than 45 minutes qualify. Refunds above "
                    "INR 1000 require human approval. All requests are within "
                    "the seven-day policy window."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "steps": [],
        "tools_executed": [],
        "llm_calls": 0,
    }

    _write_checkpoint(
        run_id, scenario_id, checkpoint["messages"], 0,
        [], [], 0,
    )

    return _continue_agent(
        run_id, checkpoint,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools,
    )


def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
):
    path = STATE_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for run {run_id}")

    checkpoint = json.loads(path.read_text(encoding="utf-8"))

    return _continue_agent(
        run_id, checkpoint,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )


# ---------------------------------------------------------------------
# Task 6: run the seven supplied scenarios and log each result
# ---------------------------------------------------------------------

def run_scenarios():
    with SCENARIOS_PATH.open(encoding="utf-8", newline="") as file:
        scenarios = list(csv.DictReader(file))

    all_results = []

    # This file represents one batch of seven scenario runs.
    RUNS_PATH.write_text("", encoding="utf-8")

    for row in scenarios:
        mock_llm.reset()

        result = run_agent(
            user_message=row["message"],
            scenario_id=row["scenario_id"],
        )

        expected_status = row["expected_outcome"]
        expected_refund = float(row["expected_refund_inr"])

        actual_refund = 0.0
        for step in result["steps"]:
            if step["tool_name"] in {"issue_refund", "escalate_to_human"}:
                actual_refund = float(step["observation"]["refund_inr"])

        record = {
            "scenario_id": row["scenario_id"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        }

        with RUNS_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=_json_default) + "\n")

        print(
            f"{row['scenario_id']}: status={result['status']}, "
            f"refund={actual_refund:.2f}, "
            f"expected_status={expected_status}, "
            f"expected_refund={expected_refund:.2f}, "
            f"calls={result['llm_calls']}, "
            f"tokens={result['tokens_used']}"
        )

        all_results.append({
            **result,
            "expected_status": expected_status,
            "expected_refund": expected_refund,
            "actual_refund": actual_refund,
        })

    return all_results

def run_resume_demo():
    """Crash after tool 2, resume, and verify no tool is repeated."""
    with SCENARIOS_PATH.open(encoding="utf-8", newline="") as file:
        s1 = next(
            row for row in csv.DictReader(file)
            if row["scenario_id"] == "S1"
        )

    run_id = f"S1_RESUME_{uuid.uuid4().hex[:8]}"
    mock_llm.reset()

    print("\n=== CRASH AND RESUME DEMO ===")

    try:
        run_agent(
            user_message=s1["message"],
            scenario_id="S1_RESUME",
            run_id=run_id,
            crash_after_tools=2,
        )
    except RuntimeError as exc:
        print(exc)

    checkpoint_path = STATE_DIR / f"{run_id}.json"
    checkpoint = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )
    tools_before_resume = list(checkpoint["tools_executed"])

    print("Tools before resume:", tools_before_resume)

    result = resume_agent(run_id)

    all_tools = [
        step["tool_name"] for step in result["steps"]
    ]

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]

    assert all_tools == expected_tools, (
        f"Unexpected tool sequence: {all_tools}"
    )
    assert result["status"] == "REFUND", result
    assert abs(128.25 - result["steps"][-1]["observation"]["refund_inr"]) < 0.01

    print("Tools across crash and resume:", all_tools)
    print("Final status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Model calls:", result["llm_calls"])
    print("Tokens used:", result["tokens_used"])

    return result


if __name__ == "__main__":
    print("=== SEVEN AGENT SCENARIOS ===")
    run_scenarios()

    print("\n=== TOKEN LIMIT DEMO: S1 ===")
    mock_llm.reset()
    with SCENARIOS_PATH.open(encoding="utf-8", newline="") as file:
        s1 = next(row for row in csv.DictReader(file)
                  if row["scenario_id"] == "S1")

    token_demo = run_agent(
        s1["message"],
        "S1_TOKEN_DEMO",
        token_limit=200,
    )
    print(token_demo["status"], token_demo["final_answer"])

    print("\n=== TIMEOUT DEMO ===")
    mock_llm.reset()
    timeout_demo = run_agent(
        s1["message"],
        "S1_TIMEOUT_DEMO",
        time_limit_seconds=0,
    )
    print(timeout_demo["status"], timeout_demo["final_answer"])
    run_resume_demo()
