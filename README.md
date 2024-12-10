# Berkeley MIDS AIArtBuddy Capstone Project

## Team Members
- Abby Purnell
- Kai Ding
- Michael Botros
- Danny Nguyen

For more information about this project, visit [Berkeley School of Information Capstone Projects](https://www.ischool.berkeley.edu/projects/2024/ai-art-buddy).

## Setup Instructions

1. Clone this repository to your local machine.

2. Create an `api_keys.py` file in the `helpers` folder with the following content:

```python
import openai
import os

def get_keys():    
    openai_key = "YOUR_OPENAI_API_KEY"
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY"
    
    openai.api_key = openai_key
    return openai_key
```

3. Replace `"YOUR_OPENAI_API_KEY"` and `"YOUR_LANGCHAIN_API_KEY"` with your actual API keys.

## Launch Application

1. Navigate to the streamlit folder:
```bash
cd streamlit
```

2. Launch the application:
```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

## Important Notes
- Make sure to add `api_keys.py` to your `.gitignore` file to prevent accidentally committing your API keys
- Keep your API keys secure and never share them publicly
- The `api_keys.py` file is required for the application to function properly
