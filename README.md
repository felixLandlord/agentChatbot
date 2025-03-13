# agentChatbot

## Overview
This is an expert chatbot assistant designed to help users with queries related to a knowledge base. The chatbot leverages advanced AI technologies such as; langgraph and langchain to create an agent that provides accurate and concise responses, enhancing user experience.

## Features
- **AI-Powered Assistance**: Utilizes Google Generative AI and Groq models
- **Hybrid Search**: Combines vector and full-text search via MongoDB Atlas
- **Dynamic Greetings**: Personalized user greetings based on context
- **Contextual Responses**: Adapts responses based on user questions and context
- **Scalable Architecture**: Designed for scalability and performance

## Prerequisites
- Python version 3.10 or higher
- Docker
- MongoDB Atlas account
- Google Generative AI API key
- Groq API key

## Setup
### Clone Repository
```zsh
git clone https://github.com/felixLandlord/agentChatbot.git
cd agentChatbot
```

### Install Dependencies
```zsh
poetry install
```

### Environment Variables
Create a `.env` file in the root directory from the `.env.example` file and fill in the required values.

### Run the Application
__For development:__
with uvicorn:
```zsh
uvicorn app.main:app --reload
```

with fastapi:
```zsh
fastapi dev
```

__For production:__
with uvicorn:
```zsh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

with fastapi:
```zsh
fastapi run app/main.py
```

__For docker:__
```zsh
docker compose up -d
```

## API Endpoints
- **/** : API docs redirect
- **/health** : API Service status
- **/agent/init** : Start chat session
- **/agent/chat** : Chat interface
- **/vector_store/upload** : Upload JSON embeddings to vector store

### stream response
example:
```zsh
curl -N -X POST \
  'host:port/agent/chat' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{"question": "summarize the documents in 2 sentences", "thread_id": "8f303d99-ba15-4e1b-aa48-6d1885b8e0ef"}' \
  --no-buffer
```