from dotenv import load_dotenv
import os
import pandas as pd
import re
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
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pprint
from langchain_core.documents import Document

# GOOGLE_FOLDER_ID = "16JNkBJlo4uUAdz3RNwfgMQ7SRl05V9tUnIBQiE5zLXY"
GOOGLE_FOLDER_ID = "196Pf4CodXHjLqlUijeBwMSd5HXM_lakkdXFovp7YgRg"

GOOGLE_CREDENTIALS_PATH = '.credentials/credentials.json'
GOOGLE_TOKEN_PATH = '.credentials/google_token.json'
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def authenticate():
    creds = None
    if os.path.exists(GOOGLE_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(GOOGLE_TOKEN_PATH, 'w') as file:
            file.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service

def load_and_split_docs_from_document_ids(document_ids):
    loader = GoogleDriveLoader(
        file_ids = document_ids,
        token_path = GOOGLE_TOKEN_PATH,
        credentials_path = GOOGLE_CREDENTIALS_PATH,
        )
    documents = loader.load()

    return documents


documents = load_and_split_docs_from_document_ids(["16JNkBJlo4uUAdz3RNwfgMQ7SRl05V9tUnIBQiE5zLXY"])

print(documents)