from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# 1. Create message history (memory)
history = ChatMessageHistory()

# 2. Prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who remembers the conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# 3. Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    
    temperature=0.7
)

# 4. Build chain
chain = prompt | llm | StrOutputParser()

# 5. Wrap with memory
chatbot = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

# 6. Chat loop
session_id = "ramkumar-session"

print("Groq Chatbot with Memory is running. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = chatbot.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    print("Bot:", response)
