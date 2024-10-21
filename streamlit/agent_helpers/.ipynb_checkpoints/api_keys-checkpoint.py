import os
import openai

openai_key = "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"
        
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LANGCHAIN_API_KEY"] = "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"

openai.api_key = openai_key