from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import InboxAction, InboxObservation, InboxState, Message


class SocialInboxEnvClient(EnvClient[InboxAction, InboxObservation, InboxState]):
    def _step_payload(self, action: InboxAction) -> dict:

        """Convert action to JSON"""

        return {"message_id": action.message_id,
                "priority": action.priority,
                "category": action.category,                
                "action": action.action}
    
    def _parse_result(self, payload: dict) -> StepResult:

        """Parse JSON to observation"""
        obs = InboxObservation(
            message=Message(**payload["message"]),
            inbox_size=payload["inbox_size"],
            current_step=payload["current_step"],
            reward=payload["reward"],
            done=payload["done"]
        )
        return StepResult(
            observation=obs,
            reward=payload['reward'],
            done=payload['done']
        )
    
    def _parse_state(self, payload: dict) -> InboxState:
        return InboxState(
            current_index=payload["current_index"],
            total_messages=payload["total_messages"],
            processed_message_ids=payload["processed_message_ids"],
            score=payload["score"],
            done=payload["done"]
        )