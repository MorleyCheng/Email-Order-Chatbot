"""RAG 聊天機器人主程式"""
import os
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
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
import openai
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

def init_session_state():
    """初始化 session state"""
    if 'messages' not in st.session_state:  # 存儲對話歷史
        st.session_state.messages = []
    if 'model_settings' not in st.session_state:  # 存儲模型設定
        st.session_state.model_settings = DEFAULT_CONFIG.copy()
    if 'show_retrieval' not in st.session_state:  # 存儲 UI 狀態
        st.session_state.show_retrieval = False

def validate_api_key(api_key: str) -> bool:
    """驗證 OpenAI API Key"""
    try:
        # 建立 OpenAI 客戶端，移除代理設定
        client = OpenAI(
            api_key=api_key,
            timeout=10.0  # 設定超時時間
        )
        
        # 測試呼叫 chat.completions API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "ping"}],
            max_tokens=5  # 限制回應長度以加快驗證速度
        )
        
        if response:
            st.success("✅ API Key 驗證成功！")
            return True
    except AuthenticationError:
        st.error("❌ API Key 驗證失敗：認證錯誤，請檢查 API Key 是否正確")
    except RateLimitError:
        st.error("❌ API Key 驗證失敗：已超過使用限制，請檢查帳戶額度")
    except APIConnectionError:
        st.error("❌ API Key 驗證失敗：無法連接到 OpenAI API，請檢查網路連線")
    except Exception as e:
        st.error(f"❌ API Key 驗證失敗")
        #st.error(f"❌ API Key 驗證失敗：{str(e)}")  #for debugging purposes
    return False

# 基本設定
BASE_DIR = Path(__file__).parent
DATA_FILE = str(BASE_DIR / "email_data.xlsx")
CHROMA_DIR = str(BASE_DIR / "chroma_db")

# 模型選項和限制
MODEL_OPTIONS = {
    "embedding_models": ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
    "chat_models": ["o4-mini","gpt-4o"]
}

MODEL_CONSTRAINTS = {
    "o4-mini": {
        "temperature": {"min": 1.0, "max": 2.0}
    },
    "gpt-4o": {
        "temperature": {"min": 0.0, "max": 1.0}
    }
}

# 預設設定
DEFAULT_CONFIG = {
    "openai": {
        "embedding_model": "text-embedding-3-large",
        "chat_model": "o4-mini",
        "temperature": 1.0
    },
    "vectordb": {
        "chunk_size": 200,
        "chunk_overlap": 20,
        "collection_name": "faq"
    },
    "retrieval": {
        "fetch_k": 100,
        "k": 30,
        "mmr_score_cache": True,
        "mmr_rerank_top_k": 5
    }
}

# 價格資料
PRICE_DICT = {
    "pricing": {
        "幻糖": 40, "甜蜜爆彈": 15, "彩虹奇緣": 25, "嘻哈跳跳糖": 30,
        "月光巧克力球": 50, "糖雲": 20, "魔法能量糖": 100, "莓果派對棒棒糖": 10,
        "驢子軟糖": 35, "甜光迷蹤": 150
    }
}

@st.cache_data
def load_data():
    """載入 Excel 資料"""
    try:
        return pd.read_excel(DATA_FILE)
    except Exception as e:
        st.error(f"載入資料失敗：{str(e)}")
        st.info(f"請確認檔案路徑：{DATA_FILE}")
        st.stop()

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
def init_vectordb(_documents, config):
    """初始化向量資料庫"""
    try:
        embedding = OpenAIEmbeddings(
            model=config["openai"]["embedding_model"],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["vectordb"]["chunk_size"],
            chunk_overlap=config["vectordb"]["chunk_overlap"],
            separators=["\n\n", "\n", "(?<=。)", "(?<=！)", "(?<=？)", ""]
        )
        chunks = text_splitter.split_documents(_documents)
        
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=CHROMA_DIR,
            collection_name=config["vectordb"]["collection_name"]
        )
        vectordb.persist()
        return vectordb
    except Exception as e:
        st.error(f"初始化向量資料庫失敗：{str(e)}")
        st.info(f"請確認資料夾權限：{CHROMA_DIR}")
        st.stop()

@st.cache_resource
def create_chain(_vectordb, config):
    """建立 QA 鏈"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            你是一位郵件查詢專家。你的工作是根據使用者的問題在 context 中找到最正確的回答。
            context 中包含的資訊：
            1. 產品價格：位於 '產品價格列表：' 之後
            2. 銷售數量：以json格式儲存在"tables_json"的rows[0][1] 欄位
            3. 產品名稱：以字典的形式儲存在"tables_json的rows[0][0] 欄位
            4. 如果提供的人名不存在, 請直接回答"不知道"
            5. 如果使用者詢問"某人"的總共購買價格, 請把所有Hi "某人"開頭的訂單明細裡的小計加總回答


            Context: {context}
            Question: {question}
            請根據上述 context 詳細回答問題。
        """
    )
    
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model=config["openai"]["chat_model"], 
            temperature=config["openai"]["temperature"],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ),
        retriever=_vectordb.as_retriever(
            search_kwargs=config["retrieval"],
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
        columns=["產品名稱", "價格 (TWD)"]
    )
    
    # 使用 Streamlit 的表格功能顯示
    st.sidebar.dataframe(
        price_df,
        column_config={
            "價格 (USD)": st.column_config.NumberColumn(
                format="$%d",
                help="產品價格（美元) "
            )
        },
        hide_index=True,
        width=300
    )

def display_model_settings():
    """在側邊欄顯示模型參數設定"""
    with st.sidebar:
        st.markdown("### ")
        with st.expander("## 模型參數設定"):
            # OpenAI 設定
            st.markdown("### OpenAI 設定")
            st.session_state.model_settings["openai"]["embedding_model"] = st.selectbox(
                "Embedding 模型",
                options=MODEL_OPTIONS["embedding_models"],
                help="選擇文字向量化模型",
                key="embedding_model_select"
            )
            
            chat_model = st.selectbox(
                "Chat 模型",
                options=MODEL_OPTIONS["chat_models"],
                help="選擇對話模型",
                key="chat_model_select"
            )
            st.session_state.model_settings["openai"]["chat_model"] = chat_model
            
            # 根據選擇的模型設定 temperature 範圍
            temp_min = MODEL_CONSTRAINTS[chat_model]["temperature"]["min"]
            temp_max = MODEL_CONSTRAINTS[chat_model]["temperature"]["max"]
            
            temperature = st.slider(
                "Temperature",
                min_value=temp_min,
                max_value=temp_max,
                value=max(temp_min, st.session_state.model_settings["openai"]["temperature"]),
                step=0.1,
                help=f"控制回答的隨機性（{chat_model} 模型範圍：{temp_min}-{temp_max}）",
                key="temperature_slider"
            )
            st.session_state.model_settings["openai"]["temperature"] = temperature
            
            # 向量資料庫設定
            st.markdown("### 向量資料庫設定")
            st.session_state.model_settings["vectordb"]["chunk_size"] = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=1000,
                value=DEFAULT_CONFIG["vectordb"]["chunk_size"],
                step=50,
                help="文件分割大小",
                key="chunk_size_input"
            )
            
            st.session_state.model_settings["vectordb"]["chunk_overlap"] = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=100,
                value=DEFAULT_CONFIG["vectordb"]["chunk_overlap"],
                step=10,
                help="文件分割重疊長度",
                key="chunk_overlap_input"
            )
            
            # 檢索設定
            st.markdown("### 檢索設定")
            st.session_state.model_settings["retrieval"]["fetch_k"] = st.number_input(
                "Fetch K",
                min_value=10,
                max_value=500,
                value=DEFAULT_CONFIG["retrieval"]["fetch_k"],
                step=10,
                help="初始檢索數量",
                key="fetch_k_input"
            )
            
            st.session_state.model_settings["retrieval"]["k"] = st.number_input(
                "K",
                min_value=1,
                max_value=50,
                value=DEFAULT_CONFIG["retrieval"]["k"],
                step=1,
                help="最終返回文件數量",
                key="k_input"
            )
            
            # 顯示重新初始化按鈕
            if st.button("套用設定", key="apply_settings"):
                st.session_state.need_reinit = True
                st.info("設定已更新，系統將重新初始化...")
                st.rerun()


def get_openai_api_key():
    """取得 OpenAI API Key"""
    with st.sidebar:
        # 先從環境變數讀取
        env_api_key = os.getenv("OPENAI_API_KEY", "")
        st.markdown("### ")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="請輸入您的 OpenAI API Key",
            value=env_api_key  # 使用環境變數的值作為預設值
        )
        
        if not api_key:
            st.warning("請輸入 OpenAI API Key 以繼續使用")
            st.stop()
        
        # 添加基本格式檢查
        if not api_key.startswith('sk-'):
            st.error("API Key 格式不正確，應該以 'sk-' 開頭")
            st.stop()
        
        # 檢查 API key 是否變更
        if 'previous_api_key' not in st.session_state or st.session_state.previous_api_key != api_key:
            # 更新環境變數
            os.environ["OPENAI_API_KEY"] = api_key
            
            # 如果有 .env 檔案，也更新它
            env_path = Path(__file__).parent / '.env'
            try:
                if env_path.exists():
                    with open(env_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 尋找並更新 OPENAI_API_KEY
                    key_updated = False
                    for i, line in enumerate(lines):
                        if line.startswith('OPENAI_API_KEY='):
                            lines[i] = f'OPENAI_API_KEY={api_key}\n'
                            key_updated = True
                            break
                    
                    # 如果沒有找到key，就新增一行
                    if not key_updated:
                        lines.append(f'OPENAI_API_KEY={api_key}\n')
                    
                    # 寫回檔案
                    with open(env_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                else:
                    # 如果 .env 檔案不存在，建立新檔案
                    with open(env_path, 'w', encoding='utf-8') as f:
                        f.write(f'OPENAI_API_KEY={api_key}\n')
            except Exception as e:
                st.warning(f"無法更新 .env 檔案：{str(e)}")
            
            st.session_state.previous_api_key = api_key
            st.session_state.need_reinit = True  # 觸發重新初始化
            st.rerun()  # 重新運行應用
                   
        return api_key

class RAGBot:
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config
        self.setup_environment()
        self.init_components()
    
    def setup_environment(self):
        """設定環境"""
        os.environ["OPENAI_API_KEY"] = self.api_key
        
    def init_components(self):
        """初始化元件"""
        with st.spinner('初始化中...'):
            self.df = load_data()
            self.documents = create_documents(self.df)
            self.vectordb = init_vectordb(self.documents, self.config)
            self.chain = create_chain(self.vectordb, self.config)
            self.qa = QA(self.chain)

def main():
    # 載入環境變數
    load_dotenv()
    
    # 設定頁面
    st.set_page_config(
        page_title="Email(訂單)查詢機器人",
        page_icon="📧",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # 自訂 CSS 樣式
    st.markdown("""
        <style>
        /* 固定輸入區域在底部 */
        .stChatInputContainer {
            position: fixed !important;
            bottom: 0 !重要;
            background-color: white !重要;
            border-top: 1px solid #ddd !important;
            z-index: 99999 !重要;
            padding: 1rem !重要;
            left: 15.625rem !重要;  /* 250px，配合側邊欄寬度 */
            right: 0 !重要;
        }
        
        /* 主要內容區域加上底部邊距，避免被固定輸入框遮擋 */
        .main-content-area {
            margin-bottom: 6rem !重要;
            padding-bottom: 2rem !重要;
        }
        
        /* 確保檢索資訊容器不會被輸入框遮擋 */
        .retrieval-info {
            margin-bottom: 5rem !重要;
        }
        </style>
    """, unsafe_allow_html=True)
    
    try:
        init_session_state()
        if 'need_reinit' not in st.session_state:
            st.session_state.need_reinit = False
        
        # 主要內容區域
        #with st.container():
            #st.markdown('<div class="main-content-area">', unsafe_allow_html=True)
            
        # 顯示標題和說明
        st.title("訂單查詢機器人")
        st.caption("""
        這是一個基於 RAG 的查詢系統, 資料來源是廠商與客戶往來的 email, 可以：
        - 查詢產品價格
        - 分析銷售數據
        - 回答相關問題
        """)
        
        # 側邊欄設定
        with st.sidebar:
            display_price_table()
            display_model_settings()
            api_key = get_openai_api_key()
            
            if not api_key or not validate_api_key(api_key):
                return
            
            # 檢索資訊開關
            st.markdown("### ")
            st.session_state.show_retrieval = st.toggle(
                '顯示檢索資訊',
                value=st.session_state.show_retrieval,
                key="retrieval_toggle"
            )
        
        # 初始化機器人
        if 'bot' not in st.session_state or st.session_state.need_reinit:
            bot = RAGBot(api_key, st.session_state.model_settings)
            st.session_state.bot = bot
            st.session_state.need_reinit = False
        else:
            bot = st.session_state.bot
        
        # 顯示範例問題
        if st.button("顯示範例問題", key="example_button"):
            st.info("範例: 哪個產品最多人買?")
        
        # 顯示對話歷史
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # 只在最後一條助理消息顯示檢索資訊
                if (message["role"] == "assistant" and 
                    idx == len(st.session_state.messages) - 1 and
                    st.session_state.show_retrieval and 
                    hasattr(bot.qa, 'last_retrieval')):
                    
                    with st.container():
                        st.markdown('<div class="retrieval-info">', unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("### 📚 檢索資訊")
                        st.write(f"找到 {bot.qa.last_retrieval['doc_count']} 份相關文件")
                        
                        tabs = st.tabs([f"文件 {i+1}" for i in range(len(bot.qa.last_retrieval['documents']))])
                        for i, tab in enumerate(tabs):
                            with tab:
                                doc = bot.qa.last_retrieval['documents'][i]
                                st.markdown("**文件內容：**")
                                st.markdown(f"```\n{doc.page_content[:500]}...\n```")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 固定輸入區域
        #with st.container():
            #st.markdown('<div class="stChatInputContainer">', unsafe_allow_html=True)
            
        # 使用者輸入處理
        if prompt := st.chat_input("請輸入您的問題..."):
            with st.chat_message("user"):
                st.write(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                response = bot.qa.query(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # 顯示檢索資訊（如果啟用）
                if st.session_state.show_retrieval and hasattr(bot.qa, 'last_retrieval'):
                    with st.container():
                        st.markdown('<div class="retrieval-info">', unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("### 📚 檢索資訊")
                        st.write(f"找到 {bot.qa.last_retrieval['doc_count']} 份相關文件")
                        
                        tabs = st.tabs([f"文件 {i+1}" for i in range(len(bot.qa.last_retrieval['documents']))])
                        for i, tab in enumerate(tabs):
                            with tab:
                                doc = bot.qa.last_retrieval['documents'][i]
                                st.markdown("**文件內容：**")
                                st.markdown(f"```\n{doc.page_content[:500]}...\n```")
            
            #st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"系統錯誤：{str(e)}")
        st.info("請重新整理頁面或聯繫系統管理員")

if __name__ == "__main__":
    main()


