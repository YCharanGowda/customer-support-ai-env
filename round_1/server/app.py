# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import os
from openai import OpenAI

from server.environment import Round1Environment, Action

app = FastAPI()
env = Round1Environment()


# ===== RESET =====
@app.post("/reset")
async def reset():
    obs = await env.reset()
    return {"observation": obs.model_dump(), "done": False}


# ===== STEP =====
class StepRequest(BaseModel):
    action: Action


@app.post("/step")
async def step(req: StepRequest):
    obs, reward, done, info = await env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


# ===== STATE =====
@app.get("/state")
async def state():
    obs = await env.state()
    return obs.model_dump() if obs else {}


# ===== LLM GENERATE =====
@app.post("/generate")
async def generate_response(req: dict):
    try:
        client = OpenAI(
            base_url=os.getenv("API_BASE_URL"),
            api_key=os.getenv("HF_TOKEN")
        )

        prompt = f"""
You are a professional customer support agent.

User issue: {req.get("issue")}
Sentiment: {req.get("sentiment")}

Write a short helpful response (1-2 lines).
"""

        completion = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        reply = completion.choices[0].message.content.strip()

        return {"response": reply}

    except Exception as e:
        return {"error": str(e)}


# ===== ROOT =====
@app.get("/")
async def root():
    return RedirectResponse(url="/web")


# ===== UI =====
@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    return """
    <html>
    <head>
        <title>Customer Support AI</title>
        <style>
            body {
                font-family: 'Segoe UI';
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }

            .container {
                background: #1e293b;
                padding: 30px;
                border-radius: 16px;
                width: 420px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            }

            input, textarea, select {
                width: 100%;
                padding: 10px;
                margin: 8px 0;
                border-radius: 8px;
                border: none;
            }

            button {
                width: 100%;
                padding: 12px;
                margin-top: 10px;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                cursor: pointer;
            }

            .reset { background: orange; }
            .step { background: limegreen; }
            .ai { background: cyan; }

            .output {
                margin-top: 10px;
                background: black;
                padding: 10px;
                border-radius: 8px;
                font-size: 12px;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h2>🤖 Customer Support AI</h2>

            <button class="reset" onclick="resetEnv()">Reset</button>

            <select id="category">
                <option value="">Category</option>
                <option value="billing">Billing</option>
                <option value="logistics">Logistics</option>
                <option value="general">General</option>
            </select>

            <select id="priority">
                <option value="">Priority</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
            </select>

            <textarea id="response" placeholder="Response"></textarea>

            <button class="ai" onclick="generateAI()">✨ Auto Generate</button>
            <button class="step" onclick="stepEnv()">Submit</button>

            <div class="output" id="output"></div>
        </div>

        <script>
            async function resetEnv() {
                let res = await fetch('/reset', {method:'POST'});
                let data = await res.json();
                document.getElementById("output").innerText = JSON.stringify(data, null, 2);
            }

            async function generateAI() {
                let state = await fetch('/state').then(r => r.json());

                let res = await fetch('/generate', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body: JSON.stringify(state)
                });

                let data = await res.json();
                document.getElementById("response").value = data.response;
            }

            async function stepEnv() {
                let category = document.getElementById("category").value;
                let priority = document.getElementById("priority").value;
                let response = document.getElementById("response").value;

                let res = await fetch('/step', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({
                        action:{category, priority, response}
                    })
                });

                let data = await res.json();
                document.getElementById("output").innerText = JSON.stringify(data, null, 2);
            }
        </script>
    </body>
    </html>
    """