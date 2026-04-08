# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Round 1 Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""
# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

"""
Customer Support AI Environment (OpenEnv Round 1)

This environment simulates real-world customer support tickets.

Agent must:
- Classify issue (category)
- Assign priority
- Generate response

Reward is based on correctness (0.0 → 1.0)
"""

from pydantic import BaseModel


# ================= MODELS =================
class Observation(BaseModel):
    ticket_id: str
    issue: str
    sentiment: str


class Action(BaseModel):
    category: str
    priority: str
    response: str


# ================= REWARD FUNCTION =================
def compute_reward(action: Action, obs: Observation) -> float:
    score = 0.0

    issue = obs.issue.lower()
    sentiment = obs.sentiment.lower()

    category = action.category.lower()
    priority = action.priority.lower()
    response = action.response.strip().lower()

    # ---- CATEGORY CHECK ----
    if "payment" in issue or "refund" in issue or "charge" in issue:
        if category == "billing":
            score += 0.4

    elif "order" in issue or "delivery" in issue:
        if category == "logistics":
            score += 0.4

    else:
        if category == "general":
            score += 0.4

    # ---- PRIORITY CHECK ----
    if sentiment in ["angry", "frustrated"]:
        if priority == "high":
            score += 0.3
    else:
        if priority in ["medium", "low"]:
            score += 0.3

    # ---- RESPONSE QUALITY ----
    if len(response.split()) >= 5:
        score += 0.3

    return round(score, 2)


# ================= ENVIRONMENT =================
class Round1Environment:
    def __init__(self, task: str = "easy"):
        self.task = task
        self.state_data = None
        self.steps = 0
        self.done = False

    # -------- RESET --------
    async def reset(self):
        if self.task == "easy":
            self.state_data = Observation(
                ticket_id="T001",
                issue="Payment failed but money deducted",
                sentiment="angry"
            )

        elif self.task == "medium":
            self.state_data = Observation(
                ticket_id="T002",
                issue="Order delayed and not delivered",
                sentiment="frustrated"
            )

        elif self.task == "hard":
            self.state_data = Observation(
                ticket_id="T003",
                issue="Multiple charges and refund not processed for weeks",
                sentiment="angry"
            )

        else:
            # fallback (important for safety)
            self.state_data = Observation(
                ticket_id="T000",
                issue="General query",
                sentiment="neutral"
            )

        self.steps = 0
        self.done = False

        return self.state_data

    # -------- STEP --------
    async def step(self, action: Action):
        # 🔥 CRITICAL FIX: ensure reset is called
        if self.state_data is None:
            await self.reset()

        self.steps += 1

        reward = compute_reward(action, self.state_data)

        # episode ends after 3 steps
        self.done = self.steps >= 3

        return self.state_data, reward, self.done, {
            "step": self.steps,
            "task": self.task
        }

    # -------- STATE --------
    async def state(self):
        return self.state_data

    # -------- CLOSE --------
    async def close(self):
        pass