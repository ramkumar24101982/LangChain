
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a funny assistant."),
    ("user", "Tell me a short joke about {topic}")
])
#llm = ChatGroq(model="llama-3.1-8b-instant",groq_api_key="YOUR_REAL_KEY",  temperature=0.7)
# Step 2: Initialize Groq LLM with API key
llm = ChatGroq(
    model="llama-3.1-8b-instant",
  
    temperature=0.7
)

# Step 3: Build chain (prompt → model → parser)
chain = prompt | llm | StrOutputParser()

# Step 4: Run the chain
result = chain.invoke({"topic": "programming"})

# Step 5: Print result
print(result)
