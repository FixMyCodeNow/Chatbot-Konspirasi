import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

st.set_page_config(page_title="Chatbot Teori Konspirasi", page_icon="🕵️")
st.title("Chatbot Teori Konspirasi")
st.caption("Menjelaskan asal-usul teori konspirasi terkenal beserta fakta-faktanya.")

INDEX_DIR = "faiss_index"

#api key

GOOGLE_API_KEY = st.sidebar.text_input("Masukkan Google API Key", type="password")

if not GOOGLE_API_KEY:
    st.warning("Masukkan Google API Key untuk melanjutkan.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

#LOAD VECTOR DATABASE
if not os.path.exists(INDEX_DIR):
    st.warning("Vector Database tidak ditemukan. Silakan jalankan build_vectorstore.py terlebih dahulu.")
    st.stop()

@st.cache_resource

def load_vectorstore(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore(GOOGLE_API_KEY)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#PROMPT TEMPLATE
system_prompt = """Kamu adalah chatbot edukasi tentang teori konspirasi di dunia
Tugas kamu adalah menjawab pertanyaan user dengan memberikan penjelasan yang jelas, ringkas, dan berbasis fakta.
1. Apa isi teori konspirasi tersebut?
2. Dari mana asal-usul teori konspirasi tersebut?
3. Fakta-fakta apa saja yang mendukung atau membantah teori konspirasi tersebut?

Selalu Bersikap Netral dan Edukatif. Jangan Membenarkan Teori Konspirasi. Jawaban harus berbasis fakta dan referensi yang jelas.

Konteks: {context}


"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

#CHAIN (pola baru LCEL)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

document_chain = create_stuff_documents_chain(llm, prompt=prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

#CHAT UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Tanyakan tentang teori konspirasi...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)
    
    with st.spinner("Mencari jawaban..."):
        try:
            result = qa_chain.invoke({"input": user_input})
            answer = result["answer"]
            sources = result["context"]
        except Exception as e:
            answer = f"Terjadi error saat mencari jawaban: {e}"
            sources = []
    
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
           with st.expander("Sumber Referensi"):
               for doc in sources:
                   st.caption(f"📄 {doc.metadata.get('source','unknown')}")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
       