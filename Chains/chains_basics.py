from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a researcher  who tells insights about {topic}."),
        ("human", "Tell me {insights_count} insights."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic": "quantum physics", "insights_count": 3})

print(result)