import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import os


from dotenv import load_dotenv

load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")

loader = TextLoader("pcs.csv")
text_docs = loader.load()

text_split = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
documents = text_split.split_documents(text_docs)

db = FAISS.from_documents(documents, OpenAIEmbeddings(api_key=open_api_key))

llm = ChatOpenAI(model_name="gpt-4o", api_key=open_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert sales assistant at a laptop store. Your job is to recommend a few laptops to the 
    customer based on their requirements.
    If the user is asking something other than buying laptops, let them know that this is a laptop store and purchasing 
    laptops is the only thing that can be done here. 
    Try to fit in the customer to a user persona like student, teacher, gamer, 
    business professional, researcher, teacher, casual everyday user, content creator etc. to recommend laptops. You 
    don't have to stick to the above personas strictly but this is a good guide. It is very important that you do not 
    mention to the user that you're trying to fit them into these personas. It is extremely important to ask user 
    questions to get more information about user's requirements and work pattern if you do not have enough 
    information to make an informed decision.It is absolutely mandatory to use only the below laptop products with 
    their detailed specification as mentioned in the inventory. Context:

<context>
{context}
</context>
         """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

history_aware_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the "
             "conversation")
])


retriever = db.as_retriever()
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, history_aware_prompt
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []


while True:
    question = input("Enter a question: ")
    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=ai_msg_1["answer"]))
    print("\nSales bot: ", ai_msg_1["answer"])
