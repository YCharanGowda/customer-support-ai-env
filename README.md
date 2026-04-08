---
title: Customer Support AI Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Round 1 Environment (OpenEnv)

This project implements a custom **OpenEnv environment** for Round 1 of the hackathon.

It includes:

* FastAPI server for environment interaction
* Docker container for deployment
* Inference script for evaluation

---
# Customer Support AI Environment (OpenEnv)

## Overview

This project implements a Customer Support AI Environment using OpenEnv.  
It simulates real-world customer support scenarios where an agent must classify issues, assign priority, and generate appropriate responses.

The environment is designed for evaluating AI agents based on decision-making, response quality, and contextual understanding.

---

## Features

- Real-world customer support simulation
- Multiple tasks with increasing difficulty (easy, medium, hard)
- Reward-based evaluation (0.0 to 1.0)
- FastAPI-based server with web interface
- LLM-based response generation
- Docker support for deployment
- OpenEnv specification compliance

---

## Environment Design

### Observation

The environment provides the following observation:

```json
{
  "ticket_id": "T001",
  "issue": "My payment failed but money was deducted",
  "sentiment": "angry"
}
```
Action

The agent must return:

{
"category": "billing",
"priority": "high",
"response": "We are sorry, your issue will be resolved soon."
}
Reward Function

The reward is computed based on:

Correct category classification: +0.4
Correct priority assignment: +0.3
Quality of response (length and clarity): +0.3

Total reward range: 0.0 to 1.0

Tasks

The environment includes three tasks:

Easy: Basic classification of issue
Medium: Requires sentiment-based reasoning
Hard: Requires full correct action and meaningful response
Setup Instructions
1. Install Dependencies
   pip install -r requirements.txt
2. Configure Environment Variables
   export API_BASE_URL=https://router.huggingface.co/v1
   export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
   export HF_TOKEN=your_token_here
   Running the Server
   uvicorn round_1.server.app:app --reload --port 8001

Open the web interface:

http://127.0.0.1:8001/web
API Endpoints
POST /reset → Reset environment
POST /step → Perform action
GET /state → Get current state
POST /generate → Generate AI response
Running Inference
export PYTHONPATH=.
python round_1/inference.py

The script outputs structured logs:

[START]
[STEP]
[END]
Docker Usage

Build the image:

docker build -t round_1-env .

Run the container:

docker run -p 8000:8000 round_1-env

Access:

http://localhost:8000/web
Evaluation
Supports multiple tasks
Deterministic reward computation
Score range: 0.0 to 1.0
Compatible with OpenEnv validation
Use Cases
Training customer support AI agents
Evaluating large language models
Benchmarking decision-making systems
Conclusion

This project provides a structured and realistic environment for evaluating 
AI systems in customer support scenarios, with clear reward mechanisms and scalable design.

