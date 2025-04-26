from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model='claude-3-5-haiku-20241022'

llm = ChatAnthropic(model=model, temperature=0.2, max_tokens=1024)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

resp = chain.invoke({"input": "how can langsmith help with testing?"})

print(resp)