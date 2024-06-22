# Text Summarization API

This project implements a REST API using Flask that performs text summarization using a language model. The API accepts a long piece of text and a maximum number of words as input and returns a concise summary of the text with the specified maximum number of words.

## Requirements

- Python 3.6 or later
- Conda (Anaconda or Miniconda)

## Setup Instructions

### 1. Create and Activate a Conda Environment

1. Create a new conda environment:
    ```bash
    conda create --name text_summarization_env python=3.8
    ```

2. Activate the conda environment:
    ```bash
    conda activate text_summarization_env
    ```

### 2. Install Dependencies

1. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Run the Flask Application

1. Start the Flask server:
    ```bash
   export FLASK_APP=app.py
   flask run
    ```

   - The server will start running at `http://127.0.0.1:5000`.

## Making a Request

You can use `curl` to make a POST request to the API. Below is an example of how to make a request to the `/summarize` endpoint:

```bash
curl -X POST http://127.0.0.1:5000/summarize -H "Content-Type: application/json" -d '{"text": "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI is becoming increasingly prevalent in various industries, transforming the way we work and live.", "max_words": 20}'curl -X POST http://127.0.0.1:5000/summarize -H "Content-Type: application/json" -d '{"text": "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI is becoming increasingly prevalent in various industries, transforming the way we work and live.", "max_words": 20}'
curl -X POST http://127.0.0.1:5000/summarize -H "Content-Type: application/json" -d '{"text": "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI is becoming increasingly prevalent in various industries, transforming the way we work and live.", "max_words": 20}'
