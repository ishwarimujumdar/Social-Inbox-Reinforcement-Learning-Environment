import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from openenv.core.env_server import Environment
from models import InboxAction, InboxObservation, InboxState, Message

class InboxTriageEnv(Environment):
    """
    RL Environment for Social Inbox Triage

    Tasks:
    - easy: basic classification of messages
    - medium: priority-sensitive handling
    - hard: Business-aware decision making with contextual penalties

    Same dataset, different reward functions.
    """

    def __init__(self, task_type='easy'):
        self.task_type = task_type

        self.messages = []
        self.labels = {}
        self.index = 0
        self.score = 0.0
        self.processed = []

        BASE_DIR = Path(__file__).resolve().parent.parent
        self.data_file = BASE_DIR / "data" / "message_data.json" 

   
    def reset(self) -> InboxObservation:
        self.messages = self._load_messages()

        self.index = 0
        self.score = 0.0
        self.processed = []

        return self._get_observation(reward=0.0, done=False)


    def step(self, action: InboxAction) -> InboxObservation:
        current_msg = self.messages[self.index]

        reward = self._grade(action, current_msg)

        self.score += reward
        self.processed.append(current_msg["id"])
        self.index += 1

        done = self.index >= len(self.messages)

        if done:
            return self._get_observation(
                reward=reward,
                done=True,
                msg=current_msg
            )

        return self._get_observation(reward=reward, done=False)

    @property
    def state(self) -> InboxState:
        return InboxState(
            current_index=self.index,
            total_messages=len(self.messages),
            processed_message_ids=self.processed,
            score=self.score,
            done=self.index >= len(self.messages),
        )

   
    def _get_observation(self, reward: float, done: bool, msg: dict = None) -> InboxObservation:
        if msg is None:
            msg = self.messages[self.index]

        return InboxObservation(
            message=Message(**msg),
            inbox_size=len(self.messages),
            current_step=self.index,
            reward=reward,
            done=done
        )


    def _load_messages(self):
        with open(self.data_file, "r") as f:
            data = json.load(f)

        inbox_messages = [{k: v for k, v in d.items() if k != 'label'} for d in data]
        self.labels = {d["id"]: d["label"] for d in data}

        return inbox_messages


    def _grade(self, action: InboxAction, msg: dict) -> float:
        if self.task_type == "easy":
            return self._grade_easy(action, msg)
        elif self.task_type == "medium":
            return self._grade_medium(action, msg)
        elif self.task_type == "hard":
            return self._grade_hard(action, msg)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")


    def _grade_easy(self, action: InboxAction, msg: dict) -> float:
        label = self.labels[msg["id"]]
        reward = 0.0

        if action.category == label["category"]:
            reward += 0.4
        if action.priority == label["priority"]:
            reward += 0.3
        if action.action == label["action"]:
            reward += 0.3

        return max(0.0, min(1.0, reward))


    def _grade_medium(self, action: InboxAction, msg: dict) -> float:
        label = self.labels[msg["id"]]
        reward = 0.0

        # Core correctness scoring
        # Priority and action are weighted higher to reflect decision importance
        if action.category == label["category"]:
            reward += 0.2
        if action.priority == label["priority"]:
            reward += 0.4
        if action.action == label["action"]:
            reward += 0.4

       
        if label["priority"] == "high":
            if action.action == "ignore":
                reward -= 0.5
            reward -= min(0.3, self.index * 0.03)   #Additional time-based penalty encourages earlier correct handling

        # Helps discourage unnecessary escalation of low-priority messages
        if action.priority == "low" and action.action == "escalate":
            reward -= 0.1

        return max(0.0, min(1.0, reward))
 

    def _grade_hard(self, action: InboxAction, msg: dict) -> float:
        label = self.labels[msg["id"]]
        reward = 0.0

        if action.category == label["category"]:
            reward += 0.3
        if action.priority == label["priority"]:
            reward += 0.3
        if action.action == label["action"]:
            reward += 0.4
        
        if action.action == "respond" and label["action"] != "respond":
            reward -= 0.2  # bias control for lazy responding

        
        if label["priority"] == "high":
            if action.action == "ignore":
                reward -= 0.5
            reward -= min(0.3, self.index * 0.03)   

            if self.index > len(self.messages) * 0.75:
                reward -= 0.2  # late handling hurts more
            
            if label["category"] == "query" and action.action =="escalate":
                reward -= 0.3  # Prevent over-escalation: not all high-priority messages require escalation
      
        
        # Mixed-intent messages require nuanced understanding
        # Misclassification leads to significant penalty
        if label["category"] == "mixed" and action.category != "mixed":
            reward -= 0.4


        # Business rule: premium users should not be ignored
        if msg.get("metadata", {}).get("user_type") == "premium":
            if action.action == "ignore":
                reward -= 0.3

        if action.priority == "low" and action.action == "escalate":
            reward -= 0.3

        return max(0.0, min(1.0, reward))
 