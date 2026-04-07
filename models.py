from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field
from typing import List, Dict


class Message(BaseModel):
    id: int                     
    content: str                
    user: str                 
    timestamp: str              
    metadata: Dict[str, str] = Field(default_factory=dict)  # Optional extra info (user_type,channel etc.)


class InboxAction(Action):
    message_id: int             
    priority: str               # low / medium / high
    category: str               # spam / complaint / query / feedback / mixed
    action: str                 # ignore / respond / escalate


class InboxObservation(Observation):
    message: Message           
    inbox_size: int             
    current_step: int           
    reward: float
    done: bool


class InboxState(State):
    current_index: int                  
    total_messages: int                
    processed_message_ids: List[int]   
    score: float                       
    done: bool                         