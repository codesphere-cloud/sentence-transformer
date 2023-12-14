# Sentence Embeddings API

## Overview

This API provides an endpoint to generate sentence embeddings using the "all-MiniLM-L6-v2" model. It's built using FastAPI and offers a simple way to get embeddings for a list of sentences.

## Requirements

- FastAPI
- Uvicorn (for serving the application)
- Sentence Transformers

## Installation

To install the necessary dependencies for this project, run the following command:

```bash
pip install fastapi uvicorn sentence-transformers
```

## Running the Server

To start the server, run:

```bash
uvicorn your_app_module:app --reload
```

Replace `your_app_module` with the name of your Python file containing the FastAPI application.

## API Endpoints

### POST `/embeddings/`

Generate embeddings for a list of sentences.

#### Request Body

- `texts`: A list of sentences for which embeddings are required.

#### Query Parameters

- `key`: The API key for authentication.

#### Response

- `embeddings`: A list of embeddings, each corresponding to a sentence in the request.

#### Example Request

```json
POST /embeddings/?key=key
Content-Type: application/json

{
    "texts": ["Hello world", "FastAPI is awesome"]
}
```

#### Example Response

```json
{
    "embeddings": [
        [0.01, 0.02, ...],  // Embedding for the first sentence
        [0.03, 0.04, ...]   // Embedding for the second sentence
    ]
}
```

## Security

The API uses an API key for basic authentication. Ensure to keep your API key confidential. You can change it using the environment variable `API_KEY`.