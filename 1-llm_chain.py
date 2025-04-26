# Chat models and prompts
# Build a simple LLM application with chat models and prompt templates
# https://python.langchain.com/docs/tutorials/llm_chain/

import pprint
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Translate the following from English into {language}"),
    ("user", "{text}")
])

prompt = prompt_template.invoke({
    "language": "Italian",
    "text": "I really want some pizza and some pasta marinara! The extra virgin olive oil is absolutely fantastic and I want to buy some for my grandmother!"
})
pprint.pprint(prompt.to_messages())

# Single invocation
# response = model.invoke(prompt)
# print(response.content)

# Streaming
for token in model.stream(prompt):
    print(token.content, end="|")
