from tenacity import retry, wait_random_exponential, stop_after_attempt

def invoke_with_retry(llm, prompt):
    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def _invoke(prompt):
        return llm.invoke(prompt)
    return _invoke(prompt)