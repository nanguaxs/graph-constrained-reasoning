import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

API_BASE_URL = "https://yunwu.ai/v1"
API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-api-key")
MODEL_NAME = "qwen3.5-122b-a10b"


@tool
def read_email(email_id: str) -> str:
    """Read an email."""
    return f"email content: {email_id}"


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"sent to {recipient}, subject={subject}"


model = ChatOpenAI(
    model=MODEL_NAME,
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

agent = create_agent(
    model=model,
    tools=[read_email, send_email],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Approval required before sending email.",
                },
                "read_email": False,
            },
            description_prefix="Pending approval",
        )
    ],
)

config = {"configurable": {"thread_id": "demo-thread-1"}}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Send an email to alice@example.com with subject "
                    "Meeting Reminder and body Please join the meeting "
                    "tomorrow at 3 PM."
                ),
            }
        ]
    },
    config=config,
    version="v2",
)

interrupts = result.get("__interrupt__", [])

if interrupts:
    print("Pending human review:")
    print(interrupts[0].value)

    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                    # Or:
                    # {"type": "reject", "message": "Do not send this email yet."}
                    # {
                    #     "type": "edit",
                    #     "edited_action": {
                    #         "name": "send_email",
                    #         "args": {
                    #             "recipient": "bob@example.com",
                    #             "subject": "Updated subject",
                    #             "body": "Updated body"
                    #         }
                    #     }
                    # }
                ]
            }
        ),
        config=config,
        version="v2",
    )

print(result["messages"][-1].content)
