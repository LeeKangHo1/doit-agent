#!/usr/bin/env python
# coding: utf-8

# In[2]:


from glob import glob

# for g in glob('../data/*.pdf'):
#     print(g)


# In[3]:


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_pdf_and_split_text(pdf_path, chunk_size=1000, chunk_overlap=100):
    """
    주어진 PDF 파일을 읽고 텍스트를 분할합니다.
    """
    print(f"PDF: {pdf_path}------------------------")

    pdf_loader = PyPDFLoader(pdf_path)
    data_from_pdf = pdf_loader.load()

    text_spiltter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )

    splits = text_spiltter.split_documents(data_from_pdf)

    print(f"Number of splits: {len(splits)}\n")
    return splits


# In[4]:


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

# 벡터db 설정
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

persist_directory='../chroma_store'

if os.path.exists(persist_directory):
    print("Loading existing Chroma store")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
    )

else:
    print("Creating new Chroma store")

    vectorstore = None
    for g in glob('../data/*.pdf'):
        chunks = read_pdf_and_split_text(g)
        # 100개씩 나눠서 저장
        for i in range(0, len(chunks), 100):
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=chunks[i:i+100],
                    embedding=embedding,
                    persist_directory=persist_directory,
                )
            else:
                vectorstore.add_documents(
                    documents=chunks[i:i+100],
                )



# In[5]:


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# chunks = retriever.invoke("서울 온실가스 저감 계획")

# for chunk in chunks:
#     print(chunk.metadata)
#     print(chunk.page_content)


# In[ ]:


from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
# model.invoke('안녕하세요!')

