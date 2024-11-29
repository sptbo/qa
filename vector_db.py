import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain.schema import Document

def initialize_vector_db(embedding_model, texts, persist_directory='./faiss_db'):
    if not texts:
        logging.warning("No texts found. Skipping vector database initialization.")
        return None

    logging.info(f"传入的嵌入模型: {embedding_model}")  # 添加日志，输出传入的信息

    vector_db = None
    try:
        # 分批处理 texts
        batch_size = 1  # 每批处理的文本数量

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logging.info(f"正在处理的每批文本内容: {batch_texts}")  # 添加日志，输出每批处理的文本内容

            # 将当前批次的文本转换为Document对象列表，确保只传入字符串类型作为page_content
            batch_documents = []
            for text in batch_texts:
                if isinstance(text, str):
                    batch_documents.append(Document(page_content=text))
                elif isinstance(text, Document):
                    batch_documents.append(text)
                else:
                    logging.warning(f"Unexpected data type in batch_texts: {type(text)}. Skipping this element.")

            if vector_db is None:
                vector_db = FAISS.from_documents(batch_documents, embedding_model)
            else:
                vector_db.add_documents(batch_documents)
                # 检查当前批次文档是否已存在于向量数据库中
                # new_documents_to_add = []
                # for doc in batch_documents:
                #     if not is_document_in_vector_db(doc, vector_db):
                #         new_documents_to_add.append(doc)

                # if new_documents_to_add:
                #     logging.info(f"正在添加的文档: {[doc.page_content for doc in new_documents_to_add]}")  # 添加日志，输出正在添加的文档信息
                #     vector_db.add_documents(new_documents_to_add)

        # 持久化FAISS中所有数据
        vector_db.save_local(persist_directory)

        # 打印数据库中的所有内容
        print_all_documents(vector_db)

        return vector_db
    except Exception as e:
        logging.error(f"Error initializing vector database: {e}")
        raise

# def is_document_in_vector_db(document, vector_db):
#     """
#     判断给定的文档是否已存在于向量数据库中
#     :param document: 要检查的文档对象（langchain.schema.Document类型）
#     :param vector_db: 向量数据库对象
#     :return: 如果文档已存在则返回True，否则返回False
#     """
    
#     logging.info(f"待检查的文档: {document.page_content}")
    
#     # 降低相似度阈值，增加返回的文档数量
#     existing_docs = vector_db.similarity_search(document.page_content, similarity_threshold=0.9, k=1)    
#     logging.info(f"数据库中找到的文档数量: {len(existing_docs)}")    
       
#     for existing_doc in existing_docs:    
#         logging.info(f"数据库中找到的文档: {existing_doc.page_content}")    
#         if existing_doc.page_content == document.page_content:
#             logging.info("Document already exists in vector database.")
#             return True
#     return False

def print_all_documents(vector_db):
    """
    打印数据库中的所有内容
    :param vector_db: 向量数据库对象
    """
    logging.info("打印数据库中的所有内容:")
    placeholder_query = "placeholder_query"  # 使用占位符查询
    all_documents = vector_db.similarity_search(placeholder_query, k=1000)  # 获取所有文档
    for doc in all_documents:
        logging.info(f"文档内容: {doc.page_content}")