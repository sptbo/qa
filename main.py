import os
import logging
import gradio as gr
from langchain_community.llms import QianfanLLMEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from config import ConfigManager
from document_loader import load_documents
from text_splitter import split_documents
from vector_db import initialize_vector_db
from qa_system import setup_qa_system, query_qa_system

def main():
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler("qa.log", encoding='utf-8')  # 文件输出，使用 UTF-8 编码
        ]
    )

    # Load configuration
    config_manager = ConfigManager()
    file_path = config_manager.get_config('file_path')
    llm_model_name = config_manager.get_config('llm_model_name')
    embedding_model_name = config_manager.get_config('embedding_model_name')
    base_url = config_manager.get_config('base_url')
    similarity_threshold = config_manager.get_config('similarity_threshold')
    summary_length = config_manager.get_config('summary_length')
    ai_length = config_manager.get_config('ai_length')

    os.environ["QIANFAN_AK"] = "JRLvQvdUbRXWtYG1hKmbrkwY"
    os.environ["QIANFAN_SK"] = "RhFqbIO1tDTEgzpEIkqHKa4LpQOQu1vq"

    logging.info(f"Using Base URL: {base_url}")
    logging.info(f"Using LLM Model: {llm_model_name}")
    logging.info(f"Using Embedding Model: {embedding_model_name}")
    logging.info(f"Loading documents from: {file_path}")

    llm = QianfanLLMEndpoint(model=llm_model_name)
    embedding_model = QianfanEmbeddingsEndpoint(model=embedding_model_name)

    # Load documents
    documents = load_documents(file_path)

    # Split documents
    texts = split_documents(documents)

    # Initialize vector database
    vector_db = None
    if texts:
        try:
            vector_db = initialize_vector_db(embedding_model, texts)
        except Exception as e:
            logging.error(f"Error initializing vector database: {e}")
            raise

    # Setup QA system
    qa_system = setup_qa_system(vector_db, llm)

    # Gradio Interface
    def clear_inputs():
        return "", ""

    def AI_output(prompt):
        result = query_qa_system(prompt, vector_db, llm, similarity_threshold, summary_length, ai_length)
        return result

    # 在页面加载时初始化输出框内容
    def on_load():
        return "欢迎使用智能问答系统。"

    with gr.Blocks() as demo:
        gr.Markdown("# 智能问答系统")
        with gr.Row():
            with gr.Column(scale=2):
                output_text = gr.HTML(label="^_^", elem_id="output_text", value=on_load())
                input_text = gr.Textbox(label="输入", lines=3, max_lines=3, elem_id="input_text")
                with gr.Row():
                    clear_button = gr.Button("清空", elem_id="clear_button")
                    submit_button = gr.Button("提交", elem_id="submit_button")

        # 绑定回车键事件
        input_text.submit(lambda x: AI_output(x), inputs=input_text, outputs=output_text)
        submit_button.click(AI_output, inputs=input_text, outputs=output_text)
        clear_button.click(clear_inputs, inputs=[], outputs=[input_text, output_text])

        # 设置输入框在页面加载时自动获得焦点
        input_text.focus(None, None, None)

        # 获取访问地址并打印
        _, local_url = demo.launch(inline=True, share=False) #, debug=True, server_name="0.0.0.0")

        print(f"可通过以下地址访问：{local_url}")
        logging.info(f"可通过以下地址访问：{local_url}")
        
if __name__ == "__main__":
    main()