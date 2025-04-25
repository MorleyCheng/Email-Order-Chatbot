import os
import pandas as pd
import streamlit as st
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 使用環境變數
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("請設定 OPENAI_API_KEY 環境變數")
    st.stop()

# 價格資料
PRICE_DICT = {
    "pricing": {
        "Wireless Keyboard": 40, "Wireless Mouse": 15, "USB-C Hub": 25, "Monitor Stand": 30,
        "Webcam": 50, "Laptop Cooler": 20, "External SSD": 100, "HDMI Cable": 10,
        "Desk Lamp": 35, "Ergonomic Chair": 150, "LED Monitor": 300, "Graphics Card": 400,
        "Mechanical Keyboard": 80, "Gaming Mouse": 45, "Soundbar": 60, "Network Router": 90,
        "Smart Speaker": 50, "Power Strip": 15, "USB Charger": 20, "Cooling Fan": 12
    }
}

@st.cache_data
def load_data():
    """載入 Excel 資料"""
    return pd.read_excel("./email_data.xlsx")

@st.cache_data
def create_documents(_df, price_dic=PRICE_DICT):
    """建立文件集合"""
    documents = []
    # 郵件文件
    for _, row in _df.iterrows():
        documents.append(Document(
            page_content=row['body'],
            metadata={col: row[col] for col in _df.columns}
        ))
    
    # 價格文件
    price_content = (f"{str(price_dic)}\n\n產品價格列表：\n" + 
                    "\n".join([f"{k}: {v}" for k, v in price_dic["pricing"].items()]))
    documents.append(Document(
        page_content=price_content,
        metadata={"type": "pricing"}
    ))
    return documents

@st.cache_resource
def init_vectordb(_documents):
    """初始化向量資料庫"""
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\n", "\n", "(?<=。)", "(?<=！)", "(?<=？)", ""]
    )
    chunks = text_splitter.split_documents(_documents)
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db",
        collection_name="faq"
    )
    vectordb.persist()
    return vectordb

@st.cache_resource
def create_chain(_vectordb):
    """建立 QA 鏈"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            你是一位郵件查詢專家。你的工作是根據使用者的問題在 context 中找到最正確的回答。
            context 中包含的資訊：
            1. 產品價格：位於 '產品價格列表：' 之後
            2. 銷售數量：位於 "數量" 欄位
            3. 產品名稱：位於 'tables_json':"rows" 第1個值
            4. 如果提供的人名不存在, 請直接回答"不知道"

            Context: {context}
            Question: {question}
            請根據上述 context 詳細回答問題。
        """
    )
    
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        retriever=_vectordb.as_retriever(
            search_kwargs={
                "fetch_k": 100,
                "k": 10,
                "mmr_score_cache": True,
                "mmr_rerank_top_k": 5
            },
            search_type="mmr"
        ),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

class QA:
    def __init__(self, chain):
        self.chain = chain

    def query(self, prompt):
        try:
            result = self.chain(prompt)
            
            # 檢索資訊暫存
            self.last_retrieval = {
                "doc_count": len(result['source_documents']),
                "documents": result['source_documents']
            }
            
            return result["result"]
        except Exception as e:
            st.error(f"查詢錯誤：{str(e)}")
            return "系統發生錯誤，請稍後再試"

def display_price_table():
    """顯示價格表格"""
    st.sidebar.markdown("### 產品價格表")
    
    # 將字典轉換為 DataFrame
    price_df = pd.DataFrame(
        list(PRICE_DICT["pricing"].items()),
        columns=["產品名稱", "價格 (USD)"]
    )
    
    # 使用 Streamlit 的表格功能顯示
    st.sidebar.dataframe(
        price_df,
        column_config={
            "價格 (USD)": st.column_config.NumberColumn(
                format="$%d",
                help="產品價格（美元）"
            )
        },
        hide_index=True,
        width=300
    )

def init_chat_interface(robot):
    """初始化聊天介面"""
    st.title("Email(訂單)查詢機器人")
    
    # 顯示價格表格
    display_price_table()
    
    # 在對話框上方加入切換按鈕
    col1, col2 = st.columns([4, 1])
    with col2:
        # 加入 CSS 樣式避免文字換行
        st.markdown("""
            <style>
            .stCheckbox > label {
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            </style>
        """, unsafe_allow_html=True)
        show_retrieval = st.toggle('顯示檢索資訊', value=False, key="retrieval_toggle")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 顯示對話歷史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if st.button("顯示範例問題"):
        st.info("範例: 哪個產品最多人買?")
    
    # 使用者輸入處理
    if prompt := st.chat_input("請輸入您的問題..."):
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response = robot.query(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # 顯示檢索資訊（如果啟用）
            if show_retrieval and hasattr(robot, 'last_retrieval'):
                st.markdown("---")
                st.markdown("### 📚 檢索資訊")
                st.write(f"找到 {robot.last_retrieval['doc_count']} 份相關文件")
                
                # 使用 tabs 替代巢狀 expander
                tabs = st.tabs([f"文件 {i+1}" for i in range(len(robot.last_retrieval['documents']))])
                for i, tab in enumerate(tabs):
                    with tab:
                        doc = robot.last_retrieval['documents'][i]
                        st.markdown("**文件內容：**")
                        st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.markdown("**Metadata：**")
                            st.json(doc.metadata)

def main():
    with st.spinner('系統初始化中...'):
        df = load_data()
        documents = create_documents(df)
        vectordb = init_vectordb(documents)
        chain = create_chain(vectordb)
        robot = QA(chain)
    
    init_chat_interface(robot)

if __name__ == "__main__":
    main()


