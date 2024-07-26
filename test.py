from dotenv import load_dotenv
import os
import pickle
import json
import uuid
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
import pprint

GOOGLE_FOLDER_ID = "1lmajwMbvbtPV6EuIPVQbVVKI7_gLTUzy"

def get_docs_from_drive(folder_id):
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
    
    return local_docs

documents = get_docs_from_drive(GOOGLE_FOLDER_ID)
pprint.pprint(documents)