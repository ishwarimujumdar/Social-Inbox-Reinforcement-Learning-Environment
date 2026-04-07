"""
Inference Script for Social Inbox Triage Task
=============================================

MANDATORY ENV VARIABLES:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

STDOUT FORMAT:
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import re
import textwrap
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.append(str(Path(__file__).parent))
from server.environment import InboxTriageEnv
from models import InboxAction

# =========================
# CONFIGURATION
# =========================
API_KEY = os.getenv("HF_TOKEN") #or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("TASK_NAME", "inbox-triage")
BENCHMARK = os.getenv("BENCHMARK", "social_inbox")
DIFFICULTY = os.getenv("DIFFICULTY", "medium")

MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent managing a social media inbox.

    For each message:
    - Category: spam | complaint | query | feedback | mixed
    - Priority: low | medium | high
    - Action: ignore | respond | escalate

    Respond ONLY in JSON:
    {
        "category": "...",
        "priority": "...",
        "action": "..."
    }
    """
).strip()


# =========================
# LOGGING
# =========================
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# =========================
# PROMPT BUILDING
# =========================
def build_user_prompt(step, message, state, last_reward, history):
    history_block = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
    Step: {step}

    CURRENT MESSAGE:
    {message.content}

    Last reward: {last_reward:.2f}

    Previous steps:
    {history_block}

    ENV STATE:
    - Progress: {state.current_index}/{state.total_messages}
    - Score: {state.score:.2f}

    Goal:
    Maximize total reward across the episode.

    Send your next action as JSON:
    {{
        "priority": "...",
        "category": "...",
        "action": "..."
    }}
    """).strip()

# =========================
# LLM CALL
# =========================
def extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM output."""
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No valid JSON found")


def get_model_decision(client, step, message, state, last_reward, history):
    user_prompt = build_user_prompt(step, message, state, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (completion.choices[0].message.content or "").strip()
        return extract_json(text)

    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return {
            "priority": "medium",
            "category": "query",      
            "action": "respond"
        }


# =========================
# MAIN LOOP
# =========================
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = InboxTriageEnv(task_type=DIFFICULTY)

    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = 0.0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            message = obs.message
            state = env.state

            decision = get_model_decision(
                client,
                step,
                message,
                state,
                last_reward,
                history
            )

            action = InboxAction(
                message_id=message.id,
                priority=decision.get("priority", "medium"),
                category=decision.get("category", "query"),
                action=decision.get("action", "respond"),
            )

            obs = env.step(action)

            reward = obs.reward or 0.0
            done = obs.done

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            action_str = (
                f"id={action.message_id} "
                f"pri={action.priority} "
                f"cat={action.category} "
                f"act={action.action}"
            )

            log_step(step, action_str, reward, done, None)

            # history (aligned with sample)
            history.append(
                f"Step {step}: {action.action} ({action.category},{action.priority}) -> reward {reward:.2f}"
            )

            if done:
                break

        # scoring
        if rewards:
            score = sum(rewards) / len(rewards)

        score = max(0.0, min(1.0, score))
        success = score > 0.1

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)

    finally:
        try:
            env.close()
        except:
            pass

        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()