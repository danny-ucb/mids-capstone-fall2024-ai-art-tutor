import openai
import os


def get_keys():    

    openai_key = "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"
    # openai_key = "sk-pSDPdsokzWEVJitrgC7BGJO3AcT0Fc3hjmTKF_5mN0T3BlbkFJzY18Pst1pR4nRk0c3Fn0Z1xzhtpGru-pj7y0VdDKoA"       
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["LANGCHAIN_API_KEY"] = "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"
    
    openai.api_key = openai_key

    return openai_key