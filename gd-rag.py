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

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

GOOGLE_FOLDER_ID = "1lmajwMbvbtPV6EuIPVQbVVKI7_gLTUzy"

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
        with open(GOOGLE_CREDENTIALS_PATH, 'w') as file:
            file.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service

def get_metadata_from_drive_recursive(service=authenticate(), folder_id=GOOGLE_FOLDER_ID):
    results = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"
    while True:
        response = service.files().list(q=query,
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name, modifiedTime, mimeType)',
                                        pageToken=page_token).execute()
        for file in response.get('files', []):
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                results.extend(get_metadata_from_drive_recursive(service, file['id']))
            else:
                results.append({
                    'google_doc_id': file['id'],
                    'title': file['name'],
                    'Updated': file['modifiedTime']
                })
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break  
    return results

def get_metadata_dataframe_from_drive(service=authenticate(), folder_id=GOOGLE_FOLDER_ID):
    results = get_metadata_from_drive_recursive(service, folder_id)
    metadata_df = pd.DataFrame(results)
    metadata_df['Updated'] = pd.to_datetime(metadata_df['Updated'])
    return metadata_df

def load_and_split_docs_from_folder_id_recursively(folder_id):

    loader = GoogleDriveLoader(
        folder_id = folder_id,
        token_path = GOOGLE_TOKEN_PATH,
        credentials_path = GOOGLE_CREDENTIALS_PATH,
        recursive = True
        )
    
    pre_split_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    split_docs = splitter.split_documents(pre_split_docs)
    
    return split_docs

def load_and_split_docs_from_document_ids(document_ids):

    loader = GoogleDriveLoader(
        document_ids= document_ids,
        token_path = GOOGLE_TOKEN_PATH,
        credentials_path = GOOGLE_CREDENTIALS_PATH,
        )
    
    pre_split_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    split_docs = splitter.split_documents(pre_split_docs)
    
    return split_docs

def create_db(documents):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embedding)
    vector_store.save_local("faiss_index")
    print(f"{vector_store.index.ntotal} document(s) added to store.")
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

    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Pass search_kwargs={"k": X} where x is the number of documents to retrieve (split_docs). Default is 4.

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

def extract_google_doc_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def list_vector_store_dataframe(vector_store):
    vector_df = convert_vector_store_to_dataframe(vector_store)
    print(vector_df)

def convert_vector_store_to_dataframe(vector_store):
    v_dict = vector_store.docstore._dict
    data_rows = []
    for k in v_dict.keys():
        source_url = v_dict[k].metadata['source']
        google_doc_id = extract_google_doc_id(source_url)
        title = v_dict[k].metadata['title']
        modified_time = v_dict[k].metadata['when']
        content = v_dict[k].page_content
        data_rows.append({"chunk_id": k, "google_doc_id": google_doc_id, "title": title, "Updated": modified_time, "content": content})
    vector_df = pd.DataFrame(data_rows)
    vector_df['Updated'] = pd.to_datetime(vector_df['Updated'])
    return vector_df

def extract_metadata_from_vectore_store_dataframe(vector_store):
    df = convert_vector_store_to_dataframe(vector_store)
    df.drop(columns=['chunk_id', 'content'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def delete_documents_from_vector_store(vector_store, google_doc_ids):
    vector_df = convert_vector_store_to_dataframe(vector_store)
    if not isinstance(google_doc_ids, list):
        google_doc_ids = [google_doc_ids]
    chunks_to_delete = []
    for google_doc_id in google_doc_ids:
        chunks_list = vector_df.loc[vector_df['google_doc_id'] == google_doc_id]['chunk_id'].tolist()
        chunks_to_delete.extend(chunks_list)
    vector_store.delete(chunks_list)

def add_documents_to_vector_store(vector_store, google_doc_ids):
    documents = load_and_split_docs_from_document_ids(google_doc_ids)
    vector_store.add_documents(documents)

def update_vector_store(vector_store):
    vector_store_df = extract_metadata_from_vectore_store_dataframe(vector_store)
    drive_df = get_metadata_dataframe_from_drive()

    new_documents_df = drive_df[~drive_df['google_doc_id'].isin(vector_store_df['google_doc_id'])]
    deleted_documents_df = vector_store_df[~vector_store_df['google_doc_id'].isin(drive_df['google_doc_id'])]
    merged = pd.merge(vector_store_df, drive_df, on='google_doc_id', suffixes=('_old', '_new'))
    updated_documents_df = merged[merged['Updated_old'] != merged['Updated_new']]

    new_documents = new_documents_df['google_doc_id'].tolist()
    deleted_documents = deleted_documents_df['google_doc_id'].tolist()
    updated_documents = updated_documents_df['google_doc_id'].tolist()

    if len(new_documents) != 0:
        add_documents_to_vector_store(vector_store, new_documents)
    if len(deleted_documents) != 0:
        delete_documents_from_vector_store(vector_store, deleted_documents)
    if len(updated_documents) != 0:
        delete_documents_from_vector_store(vector_store, updated_documents)
        add_documents_to_vector_store(vector_store, updated_documents)

    print(f"{len(new_documents)} document(s) added to store: {new_documents}")
    print(f"{len(deleted_documents)} document(s) deleted from store: {deleted_documents}")
    print(f"{len(updated_documents)} document(s) updated in store: {updated_documents}")
    
    vector_store.save_local("faiss_index")
    
if __name__ == '__main__':
    if os.path.exists('faiss_index/index.faiss') and os.path.exists('faiss_index/index.pkl'):
        vector_store = load_db()
        update_vector_store(vector_store)
    else:
        documents = load_and_split_docs_from_folder_id_recursively(GOOGLE_FOLDER_ID)
        vector_store = create_db(documents)
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