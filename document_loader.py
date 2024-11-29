import os
import logging
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)

def load_documents(directory):
    if not os.path.exists(directory):
        logging.error(f"Directory {directory} does not exist.")
        raise FileNotFoundError(f"Directory {directory} does not exist.")

    logging.info(f"开始处理目录: {directory}")  # 添加日志，输出正在处理的目录

    documents = []
    for root, _, files in os.walk(directory):
        for filename in files:
            # 对文件名中的特殊字符进行处理，这里简单替换为空格
            processed_filename = filename.replace('\u2022', ' ')

            logging.info(f"处理前文件名: {filename}, 处理后文件名: {processed_filename}")  # 添加日志，输出文件名处理前后情况

            try:
                loader = get_loader(processed_filename, root)
                documents.extend(loader.load())
                logging.info(f"Loaded file {processed_filename}")
            except Exception as e:
                logging.error(f"Error loading file {processed_filename}: {e}")
                logging.error(f"文件完整路径: {os.path.join(root, processed_filename)}")  # 添加更详细的错误日志，输出文件完整路径

    return documents

def get_loader(filename, directory):
    logging.info(f"正在处理文件名: {filename}, 所在目录: {directory}")  # 添加日志，输出正在处理的文件名和目录

    if filename.endswith('.txt'):
        logging.info(f"进入.txt文件加载分支，处理文件: {filename}")  # 添加日志，确认进入.txt文件加载分支
        return TextLoader(os.path.join(directory, filename), encoding='utf-8')
    elif filename.endswith('.pdf'):
        logging.info(f"进入.pdf文件加载分支，处理文件: {filename}")  # 添加日志，确认进入.pdf文件加载分支
        return PyPDFLoader(os.path.join(directory, filename))
    elif filename.endswith('.docx'):
        logging.info(f"进入.docx文件加载分支，处理文件: {filename}")  # 添加日志，确认进入.docx文件加载分支
        return Docx2txtLoader(os.path.join(directory, filename))
    elif filename.endswith('.xls') or filename.endswith('.xlsx'):
        logging.info(f"进入.xls/.xlsx文件加载分支，处理文件: {filename}")  # 添加日志，确认进入.xls/.xlsx文件加载分支
        try:
            return UnstructuredExcelLoader(os.path.join(directory, filename), include_sheets=True)
        except Exception as e:
            logging.error(f"Error loading file {filename}: {e}")
            logging.error(f"文件完整路径: {os.path.join(directory, filename)}")  # 添加更详细的错误日志，输出文件完整路径
            return None
    elif filename.endswith('.pptx'):
        logging.info(f"进入.pptx文件加载分支，处理文件: {filename}")  # 添加日志，确认进入.pptx文件加载分支
        return UnstructuredPowerPointLoader(os.path.join(directory, filename))
    else:
        logging.error(f"Unsupported file type: {filename}")
        logging.error(f"文件完整路径: {os.path.join(directory, filename)}")  # 添加更详细的错误日志，输出文件完整路径
        raise ValueError(f"Unsupported file type: {filename}")