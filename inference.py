"""
Inference Script for MyEnvEnv
=============================
This script runs the environment and logs steps in the required format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
from typing import List, Optional

from openai import OpenAI
from round_1.server.environment import Round1Environment, Action

# ===== LLM CLIENT =====
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 3


# ===== LOGGING =====
def log_start(task: str):
    print(f"[START] task={task} env=round_1 model=llm-agent", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ===== LLM AGENT =====
def generate_action(obs):
    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME"),
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": str(obs)}
            ]
        )

        reply = response.choices[0].message.content

    except Exception as e:
        # fallback if API fails
        reply = "We are sorry your issue will be resolved soon"

    return Action(
        category="billing",
        priority="high",
        response=reply
    )


# ===== RUN TASK =====
async def run_task(task: str):
    env = Round1Environment()
    rewards = []
    steps_taken = 0

    log_start(task)

    try:
        obs = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            action = generate_action(obs)

            obs, reward, done, _ = await env.step(action)

            rewards.append(reward)
            steps_taken = step

            log_step(step, str(action.model_dump()), reward, done, None)

            if done:
                break

        score = sum(rewards) / len(rewards)
        success = score > 0

    except Exception as e:
        log_step(steps_taken, "error", 0.0, True, str(e))
        success = False
        score = 0.0

    finally:
        await env.close()
        log_end(success, steps_taken, score, rewards)


# ===== MAIN =====
async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
