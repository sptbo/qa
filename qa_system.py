import logging
from langchain_community.llms import QianfanLLMEndpoint
from langchain.chains import RetrievalQA
import gradio as gr

from retry_decorator import invoke_with_retry

def setup_qa_system(vector_db, llm):
    try:
        if vector_db:
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
        else:
            qa = llm  # Use the LLM directly if no vector database is available
        return qa
    except Exception as e:
        logging.error(f"Error setting up QA system: {e}")
        raise

def query_qa_system(prompt, vector_db, llm, similarity_threshold=0.8, summary_length=200, ai_length=500):
    try:
        logging.info(f"传入的提示信息: {prompt}, 向量数据库: {vector_db}, 语言模型: {llm}")  # 添加日志，输出传入的关键信息
        progress = gr.Progress()
        progress(0, desc="开始处理")
        
        # 使用大模型优化prompt,使其能够更准确的从向量数据库中查找到相关内容        
        # logging.info(f"优化前的提示信息: {prompt}")
        # opt_prompt = f"""
        #     请优化以下提示词：\n{prompt}\n使其能够更准确的从向量数据库中查找到相关内容。要求：返回的内容只包含优化后的提示词，排除“根据您提供的示例和要求，以下是优化后的提示词：”等类似内容。
            
        #     示例如下：
        #         原提示词：“根据三定通知，能源标准化包括哪些内容？”
        #         正确的返回格式：“能源标准化内容依据三定通知具体涵盖哪些要素？”
        #         错误的返回格式：“基于您的需求，以下是我为您优化后的提示信息：能源标准化内容依据三定通知具体涵盖哪些要素？这样的提示信息更为精准地描述了用户需求，有助于从向量数据库中查找到相关内容。”
        #     """
        # prompt = invoke_with_retry(llm, opt_prompt)
        # logging.info(f"优化后的提示信息: {prompt}")
        
        if vector_db:
            # ***从 FAISS 数据库中查找所有相似度大于 similaruti_threshold的内容（使用余弦相似度）
            results = vector_db.similarity_search_with_score(prompt, k=30)
            logging.info(f"从向量数据库查找 {prompt} 的结果: {results}")  # 添加日志，输出查找结果情况
            # high_similarity_results = [result for result, score in results if score > similarity_threshold]
            high_similarity_results = []
            for result, score in results:
                print(f"result: {result}, score: {score}")
                if score <= similarity_threshold:
                    high_similarity_results.append(result)
            logging.info(f"相似度小于等于 {similarity_threshold} 的内容: {high_similarity_results}")
            
            # 如果从向量数据库中查找到符合要求的答案
            if high_similarity_results:
                answers = [result.page_content for result in high_similarity_results]
                # 如果answers是列表，则将其连接成一个字符串。
                if isinstance(answers, list):
                    answers = "\n".join(answers)
                logging.info(f"High similarity results: {answers}")
                
                opt_answers_prompt = f"""
                    按照与\n{prompt}\n的关系处理\n{answers}\n。
                    
                    要求如下：
                    (1)删除{answers}中与{prompt}不相关的内容;
                    (2)将剩余内容按照相关度由高到低排序，
                    (3)不要改变、润色、添加任何内容;
                    (4)仅返回处理后的内容，不要添加任何解释、评论或其他无关内容。如：不要返回“以下是重排序后的内容”等类似内容；
                    (5)不要返回排序前的内容。
                    """

                answers = invoke_with_retry(llm, opt_answers_prompt)
                logging.info(f"重新排序后的答案：\n{answers}\n")
                
                answers_to_llm = answers  # 初始化 answers_to_llm

            else:
                logging.info(f"No suitable answer found in vector database with similarity <= {similarity_threshold}.")
                summary = "无相关内容。"
                answers = "无相关内容。"
                answers_to_llm = "无相关内容。"

            # ***获取summarya
            summary_prompt = f"系统任务：对 {answers} 进行总结，限定字数在 {summary_length} 字以内。仅返回总结后的内容，不要返回原文，也不要添加任何解释、评论或其他无关内容。"
            
            if answers != "无相关内容。":
                progress(0.05)
                summary = invoke_with_retry(llm, summary_prompt)
                logging.info(f"对资料的总结：{summary}")
            else:
                summary = "无相关内容。" 
        
            # ***AI根据base_answer回答。
            progress(0.45)
            if answers_to_llm != "无相关内容。":
                llm_prompt = f"请基于以下基础答案 {answers_to_llm} 回答 {prompt}，生成不超过 {ai_length} 字的内容，无需起始语与结束语。若生成答案与 {prompt} 关联度低，则以 {prompt} 为核心重新创作。"
                llm_answer = invoke_with_retry(llm, llm_prompt)
                logging.info(f"LLM Answer with answers_to_llm: {llm_answer}")

            # 如果没有从向量数据库中查找到符合要求的答案
            else:      
                llm_answer = invoke_with_retry(llm, prompt)
                logging.info(f"LLM Answer without answers_to_llm: {llm_answer}")

            # 优化显示格式
            # answers = answers.replace("\n", "")
            output_text = f"""
                <div style="font-family: Arial, sans-serif; border: 2px solid #ccc; padding: 10px; margin-bottom: 20px;">
                    <h3 style="margin-top: 20px;"> </h3>
                        <p style="margin-left: 2em; white-space: pre-line;">{answers.splitlines()[0]}</p>
                        <p style="margin-left: 2em; white-space: pre-line;">{"<br>".join(answers.splitlines()[1:])}</p>
                    <h3 style="margin-top: 0;">资料总结</h3>
                        <p style="margin-left: 2em; white-space: pre-line;">{summary.splitlines()[0]}</p>
                        <p style="margin-left: 2em; white-space: pre-line;">{"<br>".join(summary.splitlines()[1:])}</p>
                    <h3 style="margin-top: 20px;">智能补充</h3>
                        <p style="margin-left: 2em; white-space: pre-line;">{llm_answer.splitlines()[0]}</p>
                        <p style="margin-left: 2em; white-space: pre-line;">{"<br>".join(llm_answer.splitlines()[1:])}</p>
                </div>
            """
            progress(1, desc="处理完成")
            return output_text
        
        else:
            logging.info("No vector database available. Using LLM directly.")
            progress(0.75, desc="直接调用大模型")
            result = invoke_with_retry(llm, prompt)
            logging.info(f"直接调用大模型的结果: {result}")  # 添加日志，输出直接调用大模型的结果
            
            progress(1, desc="处理完成")
            return result
        
    except Exception as e:
        logging.error(f"Error invoking QA system: e")
        return "Error: Unable to process the query."