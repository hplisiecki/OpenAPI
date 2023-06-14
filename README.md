# OpenA(P)I - Threading for OpenAI API 

![OpenA(p)I logo](https://drive.google.com/file/d/1QWoRW-xS1weu_--X2YPR2_ZxGfL_ADPS/view?usp=sharing) ![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

OpenApI is a minimalistic Python package that facilitates threaded API calls to OpenAI. The package is designed to optimize the execution time for a batch of requests, making it particularly useful for fetching chat completions and text embeddings in bulk. Leveraging asyncio, the package executes API requests in parallel, providing an efficient and streamlined experience.

## Table of Contents
1. [Installation](#installation)
2. [Features](#features)
3. [Usage](#usage)
    - [Authentication](#authentication)
    - [Getting One Chat Completion](#getting-one-chat-completion)
    - [Processing Multiple Chat Completions](#processing-multiple-chat-completions)
    - [Retrieving Embeddings](#retrieving-embeddings)
4. [Dependencies](#dependencies)
5. [License](#license)
6. [Contributing](#contributing)

## Installation

I would be nice if you could install using pip, but until then:

```
git clone https://github.com/AdamSobieszek/openapi
```

## Features
- Easy authentication with the OpenAI API.
- Start chat conversations with OpenAI's GPT models.
- Process multiple chat completions in parallel.
- Retrieve embeddings for texts asynchronously and efficiently.
- Has a GPT-4 written ReadME 

## Usage

### Authentication

If you will make multiple API calls, you can set your api key to an enviroment variable and not specify it in the other functions. You can do this by providing the API key directly or by specifying a file path that contains the API key.

```python
from openapi import auth

# Authenticate by directly providing the API key
auth(api_key="your-openai-api-key")

# OR Authenticate by specifying the file path containing the API key
auth(key_path="path_to_api_keys.json")
```

### Getting One Chat Completion

You can initiate a chat conversation with an OpenAI GPT model by providing a prompt and a system message that guides the model's behavior.

```python
from openapi import call

prompt = "Tell me a joke"
system_message = "You are a friendly AI language model."
response = call(prompt, system_message, model="gpt-3.5-turbo-0613", as_str=True)

print(response)
```

### Processing Multiple Chat Completions

This package allows for processing multiple chat completions in parallel, saving the results to a file. This is especially useful for batch processing of various prompts and system messages.

```python
from openapi import chat

# Provide multiple prompts and system messages
prompts = ["Tell me a joke", "What is AI?"]
system_messages = ["You are a friendly AI language model.", "You are an educational AI."]

# Save the results to a file
chat(prompts, system_messages, save_filepath="results.json", model="gpt-3.5-turbo", api_key="your-openai-api-key")
```

### Retrieving Embeddings

You can also retrieve embeddings for given texts. The package allows for asynchronous processing in parallel.

```python
from openapi import get_embedding

texts = ["Hello, world!", "OpenAI rocks!"]
embeddings = get_embedding(text

s, save_filepath="embeddings.json", api_key="your-openai-api-key")

print(embeddings)
```

## Dependencies
- Python 3.7 or later
- openai, tiktoken, asyncio Python libraries

## License
`openapi` is licensed under the [MIT License](./LICENSE).

## Contributing
We welcome contributions! Feel free to submit a Pull Request or create an issue for any bugs or feature requests.
