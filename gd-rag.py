from dotenv import load_dotenv
import os
import pickle
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_community import GoogleDriveLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def get_docs(folder_id):
    if os.path.exists("./docs.pkl"):
        with open('./docs.pkl', 'rb') as file:
            docs = pickle.load(file)
        return docs
    
    loader = GoogleDriveLoader(
        folder_id = folder_id,
        token_path = "./.credentials/google_token.json",
        credentials_path = "./.credentials/credentials.json",
        recursive = True
        )
    pre_split = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000
    )

    docs = splitter.split_documents(pre_split)

    with open('docs.pkl', 'wb') as file:
        pickle.dump(docs, file)

    return docs

def create_db(docs):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists("./faiss_index"):
        vector_store = FAISS.load_local("./faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
        return vector_store
    vector_store = FAISS.from_documents(docs, embedding=embedding)
    vector_store.save_local("faiss_index")
    return vector_store

def create_chain(vector_store):
    model = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature= 0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context. Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Pass search_kwargs={"k": X} where x is the number of documents to retrieve (split_docs). Default is 5.

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to get information relevant to the conversation.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm = model,
        retriever = retriever,
        prompt = retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"]

if __name__ == '__main__':
    google_folder_id = "1veOSfFZDlMbB-KwSKad2BLfEfKAAJmmU"
    docs = get_docs(google_folder_id)
    vector_store = create_db(docs)
    chain = create_chain(vector_store)
    chat_history = []

    while True:
        user_input = input(f"{RED}You: {RESET}")
        if user_input.lower() == "exit":
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print(f"{GREEN} Assistant: {response}")

