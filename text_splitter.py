from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import ConfigManager

def split_documents(documents, chunk_size=100, chunk_overlap=20):
    config_manager = ConfigManager()

    # 获取chunk_size和chunk_overlap的值
    chunk_size = config_manager.get_config('chunk_size')
    chunk_overlap = config_manager.get_config('chunk_overlap')
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", 
                    "\n", 
                    "。", 
                    " ", 
                    "  ", 
                    "   ", 
                    "    "]
        )
    return text_splitter.split_documents(documents)