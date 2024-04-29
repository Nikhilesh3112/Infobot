# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

load_dotenv()


def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    # text_splitter = RecursiveCharacterTextSplitter()
    # document_chunks = text_splitter.split_documents(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(document)
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_vectorstore_from_pdf():
    # get the text in document form
    # loader = WebBaseLoader(url)
    # document = loader.load()

    text = ""
    # Open the PDF file
    with open('scheme.pdf', 'rb') as file:
        # Create a PdfReader object
        reader = PdfReader(file)

        # Loop through each page
        for i in range(len(reader.pages)):
            # Get the page object
            page = reader.pages[i]

            # Extract text from the page
            text += page.extract_text()

        # Print the page number and text
        # print(f"Page {i+1}:\n{text}\n")

        # print(text)
    # split the document into chunks
    # text_splitter = RecursiveCharacterTextSplitter()
    # document_chunks = text_splitter.split_documents(document)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the text into chunks
    text_chunks = text_splitter.split_text(text)

    # Initialize an empty list to store embeddings for each chunk
    embeddings = []

    # Generate embeddings for each chunk
    for chunk in text_chunks:
        # Assuming OpenAIEmbeddings() is the correct way to obtain embeddings
        embedding = OpenAIEmbeddings().embed(chunk)
        embeddings.append(embedding)

    # Create a vectorstore from the chunks and embeddings
    vector_store = Chroma.from_documents(text_chunks, embeddings)

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']


# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Government Info Bot")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        # st.session_state.vector_store = get_vectorstore_from_pdf()

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
