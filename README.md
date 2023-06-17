# Berlin LLM Hackathon
Code for the Berlin LLM Hackathon organized by Weaviate. Evaluating multiple retreival strategies for a Q&amp;A bot using MLOps slack messages



Make sure to create an .env file with your environement variable

```bash
OPENAI_API_KEY=sk-xxxx
COHERE_API_KEY=xxx
WEVIATE_URL=https://xxxx.weaviate.network
OPENAI_ORGANIZATION=org-xxxx
```


## Local setup

### Create environment
Create a new virtual environment and install the requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download the data
```bash
python download_data.py
```

### Run the ingestion pipeline
```bash
python ingest.py
```

## Running locally
```bash
python app.py "What is the best way to get started with MLOps?"
```

## Running in agenta

### Installing Agenta

You need to install the agenta platform locally. You can find the instructions in the github repository:

https://github.com/agenta-ai/agenta

and in the documenation

https://docs.agenta.ai

### Running the app in the agenta web-ui

```bash
agenta variant serve
```

This deploys the app locally. You can access the api here: http://localhost/llm_hackathon/original/ with its documentation in http://localhost/llm_hackathon/original/docs

You can experiment with your app in the playground. 🎮 Go to: http://localhost:3000/apps/llm_hackathon/playground