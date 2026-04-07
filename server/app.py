import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from openenv.core.env_server import create_fastapi_app
from server.environment import InboxTriageEnv  
from models import InboxAction, InboxObservation


def main():
    return create_fastapi_app(
        InboxTriageEnv,
        action_cls=InboxAction,
        observation_cls=InboxObservation
    )

app = main()


if __name__ == "__main__":
    main()