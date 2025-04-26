
from rag_agent import RagAgent

# Config

topic = 'The Company "Outrival"'

embeddings_model = "text-embedding-004"
webpage_documents = (
    'https://outrival.com/',
    'https://outrival.com/company',
    'https://outrival.com/blog/ai-for-the-people'
)

llm_model_provider = "google_genai"
llm_model = "gemini-2.0-flash"

# Interactive Chat

def main():
    """
    Prompts the user for a question and provides a simple response.
    """
    rag_agent = RagAgent(
        topic=topic,
        llm_model=llm_model,
        llm_model_provider=llm_model_provider,
        embeddings_model=embeddings_model,
        webpage_documents=webpage_documents
    )

    while True:
        user_question = input("> ")
        if user_question.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        for event in rag_agent.ask(user_question):
            event["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()
