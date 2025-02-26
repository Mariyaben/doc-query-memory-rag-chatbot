# doc-query-memory-rag-chatbot

git clone https://github.com/Mariyaben/doc-query-memory-rag-chatbot.git
cd doc-query-memory-rag-chatbot

# set up virtual env
python -m venv venv
source venv/bin/activate  #On Mac/Linux
venv\Scripts\activate     #On Windows

# install dependencies and run the chatbot
pip install -r requirements.txt
OPENAI_API_KEY=openai-api-key  #set openai api key in .env
python ingest.py
streamlit run app.py


