import anthropic
from dotenv import load_dotenv

load_dotenv()

resp = anthropic.Anthropic().messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, world"}
    ]
)

print(resp)
