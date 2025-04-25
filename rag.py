import os
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
import gradio as gr

openai_api_key = "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"
os.environ['OPENAI_API_KEY'] = "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" # arthurzllu

# 1. 讀取檔案
#df = pd.read_excel("./outlook_search.xlsx")
df = pd.read_excel("./email_data.xlsx")
#df_content = pd.read_csv("./all_train_course_contents.csv")

# 假設兩個檔案都有 '課程敘述' 欄位，我們先合併
#df = pd.merge(df_content, df_enroll, on = "課程名稱", how = "left")



from langchain.schema import Document

documents = []
for _, row in df.iterrows():
    # 使用所有欄位作為 metadata
    metadata = {col: row[col] for col in df.columns}

    # 將所有欄位的值合併為文本內容
    page_content = row['body']

    documents.append(Document(
        page_content=page_content,
        metadata=metadata
    ))

price_dic = {"pricing":{
    "Wireless Keyboard": 40, "Wireless Mouse": 15, "USB-C Hub": 25, "Monitor Stand": 30,
    "Webcam": 50, "Laptop Cooler": 20, "External SSD": 100, "HDMI Cable": 10,
    "Desk Lamp": 35, "Ergonomic Chair": 150, "LED Monitor": 300, "Graphics Card": 400,
    "Mechanical Keyboard": 80, "Gaming Mouse": 45, "Soundbar": 60, "Network Router": 90,
    "Smart Speaker": 50, "Power Strip": 15, "USB Charger": 20, "Cooling Fan": 12}}

page_content = str(price_dic) + "\n\n" + "產品價格列表：" + "\n".join([f"{k}: {v}" for k, v in price_dic["pricing"].items()])
#documents.append(Document(page_content=page_content, metadata={"pricing": str(price_dic["pricing"])}))


#print(documents)



# 3. 切分文件（避免過長）

from langchain.text_splitter import RecursiveCharacterTextSplitter

chunk_size = 200
chunk_overlap = 20

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    separators = ["\n\n", "\n", "(?<=。)", "(?<=！)", "(?<=？)", "", "(?<=.)", "(?<=!)", "(?<=?)"]
    # separators = [ "\n\n", "\n", "(?<=\。 )", " ", "" ]
)

chunks = text_splitter.split_documents(documents)


# 4. 建立 Embedding 與 VectorStore

embedding = OpenAIEmbeddings(
    model = "text-embedding-3-large", # text-embedding-ada-002,
    openai_api_key = openai_api_key
)

import chromadb
from chromadb.config import Settings

# 設定 ChromaDB
chroma_settings = Settings(
    anonymized_telemetry=False,
    is_persistent=True
)

# 刪除舊的資料庫（如果存在）
import shutil
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

collection_name = "faq"
persist_path = "./chroma_db"

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_path,
    collection_name=collection_name,
    client_settings=chroma_settings
)

# 將向量寫入硬碟
vectordb.persist()
vectordb._collection.count()


# 5. 測試 vector db

query = "uther"
results = vectordb.similarity_search(query, k = 10)

# 顯示查詢結果
for doc in results:
    print(f"📌 寄件人: {doc.metadata['sender']}")



# 6. 建立 LLM

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

model = ChatOpenAI(model = "gpt-4o", temperature = 0)



# 7. 建立 prompt

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
       input_variables=["context", "question"],
       template="""
               你是一位郵件查詢專家。你的工作是根據使用者的問題在 context 中找到最正確的回答。
               **如果 context 中沒有一樣的人名，請直接回答 "我不知道"**。
               context 中包含 pricing 信息，位於 '產品價格列表：' 之後，格式為 '產品名稱: 價格'。你需要解析它來回答問題。
               context 中包含 銷售數量 信息，位於 "數量" 你需要解析它來回答問題。
               context 中包含 產品名稱 信息，位於 'tables_json':"rows" 第1個值 你需要解析它來回答問題。


               Context: {context}
               Question: {question}
               Answer:
           """
   )

# 7. 建立 prompt

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
       input_variables=["context", "question"],
       template="""
               你是一位郵件查詢專家。你的工作是根據使用者的問題在 context 中找到最正確的回答。



               Context: {context}
               Question: {question}
               Answer:
           """
   )

# 8. 建立 QA

retriever = vectordb.as_retriever(search_kwargs = {
            "fetch_k" : 800,  #初始檢索的文件數量 (5 篇)。
            "k" : 100,    #最終返回的文件數量 (3 篇)
            "mmr_score_cache" : True,
            "mmr_rerank_top_k" : 3  #MMR 重新排序的前幾名文件數量 (10 篇)。
        }, retriever_mode = "reduce_op_recursive", search_type = "mmr")


chatbot = RetrievalQA.from_chain_type(
            llm = model,
            retriever = retriever,
            return_source_documents = True,
            chain_type = "stuff",
            chain_type_kwargs = { "prompt" : prompt }
        )


class QA():

    def __init__(self):
        self.model = RetrievalQA.from_chain_type(
            llm = model,
            retriever = retriever,
            return_source_documents = True,
            chain_type = "stuff",
            chain_type_kwargs = { "prompt" : prompt }
        )

    def query(self, prompt):
        return self.model(prompt)["result"]

robot = QA()

# 儲存對話歷史
chat_history = []

def qa_bot(user_input):
    global chat_history

    response = robot.query(user_input)

    # 將使用者輸入加入歷史
    chat_history.append({"使用者": user_input})

    # 加入機器人回應到對話歷史
    chat_history.append({"機器人": response + "\n" + "-" * 20 })

    # 格式化對話歷史輸出
    chat_display = "\n".join(
        [f"使用者: {item['使用者']}" if '使用者' in item else f"機器人: {item['機器人']}" for item in chat_history]
    )

    return chat_display

with gr.Blocks() as app:
    gr.Markdown("Email(訂單)查詢機器人")
    chatbot = gr.Textbox(label = "對話紀錄", interactive = False, lines = 10)
    user_input = gr.Textbox(label = "請輸入您的問題", placeholder = "在這裡輸入您的問題...")
    submit_button = gr.Button("送出")
    examples = [ "哪個產品最多人買?" ]
    submit_button.click(qa_bot, inputs = [user_input], outputs = [chatbot])

app.launch(server_name = "0.0.0.0", server_port = 9001, debug = True)


user_input = "keyboard賣了多少個"
response = chatbot(user_input)
print(response["result"])

user_input = "哪個產品最多人買"
response = chatbot(user_input)
print(response["result"])

user_input = "哪個產品最便宜"
response = chatbot(user_input)
print(response["result"])

user_input = "Ben在4/29有買東西嗎? 買了什麼?"
response = chatbot(user_input)
print(response["result"])

user_input = "PO-20250427-706內容?"
response = chatbot(user_input)
print(response["result"])

user_input = "Ricky在4/29有買東西嗎? 買了什麼?"
response = chatbot(user_input)
print(response["result"])