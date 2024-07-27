from dotenv import load_dotenv
import os
import pickle
import json
import uuid
import pprint
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
from langchain_core.load import dumpd, dumps, load, loads

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

GOOGLE_FOLDER_ID = "1lmajwMbvbtPV6EuIPVQbVVKI7_gLTUzy"

def get_local_documents():
    pass

def get_docs_from_drive(folder_id):
    if os.path.exists('docs.json'):
        with open('docs.json', 'r') as file:
            local_docs = loads(json.load(file))
        return local_docs

    local_docs = {}

    loader = GoogleDriveLoader(
        folder_id = folder_id,
        token_path = "./.credentials/google_token.json",
        credentials_path = "./.credentials/credentials.json",
        recursive = True
        )
    
    pre_split_docs = loader.load()
    for i, document in enumerate(pre_split_docs):
        document.id = i
        local_docs[document.id] = {
            'modified_time': document.metadata['when'],
            'splits': []
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000
    )

    for document in pre_split_docs:
        split_doc = splitter.split_documents([document])
        for split in split_doc:
            split.id = str(uuid.uuid4())
            local_docs[document.id]['splits'].append(split)

    string_representation = dumps(local_docs)
    with open('docs.json', 'w') as file:
        json.dump(string_representation, file)
    
    return local_docs

def create_db(docs):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(docs, embedding=embedding)
    vector_store.save_local("faiss_index")
    return vector_store

def load_db():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local("./faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
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
    docs = get_docs_from_drive(GOOGLE_FOLDER_ID)
    all_splits = []
    for item in docs.values():
        for split in item['splits']:
                all_splits.append(split)
    vector_store = create_db(all_splits)
    pprint.pprint(docs['0']['splits'][0].id)
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

