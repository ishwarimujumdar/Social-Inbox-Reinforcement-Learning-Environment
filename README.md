--- 
title: social-inbox 
colorFrom: blue 
colorTo: purple 
sdk: docker 
---

# Social Inbox Reinforcement Learning Environment

## Environment Description and Motivation

The Social Inbox RL Environment is designed to simulate a real-world social media inbox where messages must be triaged efficiently. Agents interact with the environment to classify incoming messages, assign priorities, and choose appropriate actions (respond, escalate, ignore). This setup encourages the development of reinforcement learning policies that balance correctness, efficiency, and business logic considerations.

**Motivation:**

* Automate inbox triage for social media support teams.
* Explore RL approaches for decision-making under uncertainty.
* Build LLM-based or agent-driven automation policies for practical NLP tasks.


## Action and Observation Space Definitions

**Action Space:**

* `category`: The type of message (`spam`, `complaint`, `query`, `feedback`, `mixed`).
* `priority`: Message urgency (`low`, `medium`, `high`).
* `action`: How the agent handles the message (`ignore`, `respond`, `escalate`).

**Observation Space:**

* `message`: The content of the current message.
* `inbox_size`: Total messages in the current episode.
* `current_step`: Index of the message being processed.
* `reward`: Reward received from the previous step.
* `done`: Whether the episode has ended.

## Task Descriptions with Expected Difficulty

| Task   | Description                                                                                                                       | Difficulty |
| ------ | --------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Easy   | Basic classification of messages. Reward for correct category, priority, and action.                                              | Easy       |
| Medium | Priority-sensitive handling. Penalizes ignoring high-priority messages and unnecessary escalations.                               | Medium     |
| Hard   | Business-aware decision making. Penalizes critical mistakes (ignoring premium users, misclassifying mixed intent, late handling). | Hard       |

## Setup and Usage Instructions


## Action and Observation Space Definitions

**Action Space:**

* `category`: Type of message (`spam`, `complaint`, `query`, `feedback`, `mixed`)
* `priority`: Urgency (`low`, `medium`, `high`)
* `action`: Handling choice (`ignore`, `respond`, `escalate`)

**Observation Space:**

* `message`: Content of the current message
* `inbox_size`: Total messages in the current episode
* `current_step`: Index of the message being processed
* `reward`: Reward received from the previous step
* `done`: Whether the episode has ended

## Task Descriptions with Expected Difficulty

| Task   | Description                                                                                                                       | Difficulty |
| ------ | --------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Easy   | Basic classification of messages. Reward for correct category, priority, and action.                                              | Easy       |
| Medium | Priority-sensitive handling. Penalizes ignoring high-priority messages and unnecessary escalations.                               | Medium     |
| Hard   | Business-aware decision making. Penalizes critical mistakes (ignoring premium users, misclassifying mixed intent, late handling). | Hard       |

## Setup and Usage Instructions

### Prerequisites

* Python >= 3.9
* Docker (for containerized environment)
* Virtual environment recommended

### Installation

```bash
# Clone the repo
git clone https://github.com/ishwarimujumdar/Social-Inbox-Reinforcement-Learning-Environment.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Inference

```bash
python inference.py
```

### Validating the Environment

```bash
openenv validate
```

## Baseline Scores

* Easy Task: ~0.9 success rate
* Medium Task: ~0.8 success rate
* Hard Task: ~0.6 success rate

