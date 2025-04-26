# Custom adaptation of https://python.langchain.com/docs/tutorials/llm_chain/

import pprint
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in the history of spaceflight"),
    ("user", "{text}")
])

prompt = prompt_template.invoke({
    "text": "Tell me about the internal space station"
})
pprint.pprint(prompt.to_messages())
print()

# Streaming
for token in model.stream(prompt):
    print(token.content, end="")
