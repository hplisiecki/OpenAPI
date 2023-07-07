import asyncio
import json
from typing import List
import openai
import os
import json
import logging
import time
try:
    # Try a relative import (when run as part of a package)
    from .api_threading import execute_api_requests_in_parallel
    from .functions import generate_schema_from_function
except ImportError:
    # Fall back to an absolute import (when run as a standalone script)
    from api_threading import execute_api_requests_in_parallel
    from functions import generate_schema_from_function


def auth(api_key=None, key_path=None):
    """
    Sets up authentication for the OpenAI API by setting API keys either
    directly from the provided `api_key` parameter, or by reading them from a file.

    Parameters:
        api_key (str, optional): Direct API key for authentication.
        key_path (str, optional): Path to the file containing the API key. If None,
                                  attempts to use a default path.

    Raises:
        Exception: If authentication fails due to inability to access the key file.
    """
    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
        return

    if key_path is None:
        key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_keys.json')

    try:
        with open(key_path) as f:
            keys = json.load(f)
            openai.api_key = keys["openai_api_key"]
            openai.organization = keys["openai_organization"]
            os.environ["OPENAI_API_KEY"] = keys["openai_api_key"]
            os.environ["OPENAI_ORGANIZATION"] = keys["openai_organization"]
    except:
        raise Exception("Authentication failed")


def call(prompt, system_message, model="gpt-3.5-turbo-0613", as_str=False):
    """
        Initiates a chat conversation with an OpenAI GPT model by sending a prompt
        along with a system message to guide the model's behavior. It can return
        the whole chat object or just the text message based on the `text` parameter.

        Parameters:
            prompt (str): The prompt or input message to send to the model.
            system_message (str): Message that sets the behavior of the model.
            model (str, optional): Model name to use. Default is "gpt-3.5-turbo-0613".
            as_str (bool, optional): If True, returns only the text message, otherwise
                                   returns the entire chat object. Default is False.

        Returns:
            Chat object or string message based on the `as_str` parameter.
        """
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    chat = openai.ChatCompletion.create(model=model, messages=messages)
    return chat if not as_str else chat.choices[0]['message']['content']



def chat_strings(prompts, system_messages, model="gpt-3.5-turbo-0613", temperature=1, top_p=1, n=1,
                 stream=False, stop=None, max_tokens=None, presence_penalty=0, frequency_penalty=0,
                 functions=None, function_call="none"):
    """
        Prepares a list of chat strings in JSON format for batch processing. Each string represents a separate chat prompt for the OpenAI API.

        Parameters:
            prompts (List[str]): List of prompts.
            system_messages (List[str]): List of system messages to guide the model's behavior.
            model (str, optional): Model name to use. Default is "gpt-3.5-turbo-0613".
            temperature (float, optional): Sampling temperature. Default is 1.
            top_p (float, optional): Nucleus sampling parameter. Default is 1.
            n (int, optional): Number of chat completion choices to generate. Default is 1.
            stream (bool, optional): Whether to send partial message deltas. Default is False.
            stop (str or List[str], optional): Up to 4 sequences where the API will stop generating further tokens.
            max_tokens (int, optional): Maximum number of tokens to generate.
            presence_penalty (float, optional): Penalty for new tokens based on their presence so far.
            frequency_penalty (float, optional): Penalty for new tokens based on their frequency so far.
            functions (List[dict], optional): List of functions defined in the OpenAI schema.
            function_call (str, optional): The type of function call to make. Default is "none".

        Returns:
            List[str]: A list of request strings in JSON format.
        """

    params = {"temperature": temperature,
              "top_p": top_p,
              "n": n,
              "stream": stream,
              "stop": stop,
              "max_tokens": max_tokens,
              "presence_penalty": presence_penalty,
              "frequency_penalty": frequency_penalty,
              "functions": functions,
              "function_call": function_call}

    default_values = {"temperature": 1, "top_p": 1, "n": 1, "stream": False, "stop": None, "max_tokens": None, "presence_penalty": 0, "frequency_penalty": 0, "functions": None, "function_call": "none"}

    jobs = [{"model": model,
             "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
             **{param: value for param, value in params.items() if value != default_values[param]}}
            for prompt, system_message in zip(prompts, system_messages)]

    return [json.dumps(job, ensure_ascii=False) for job in jobs]




def chat(prompts, system_messages, save_filepath = "chat_temp.txt", model="gpt-3.5-turbo-0613", api_key=None, verbose = True, as_str = False, **kwargs):
    """
    Processes a list of chat prompts in parallel, saving the results to a specified file.

    Parameters:
        prompts (List[str] or str): List or single prompt to send to the model.
        system_messages (List[str] or str): List or single system message to guide the model's behavior.
        save_filepath (str, optional): Path to save the results. Default is "chat_temp.txt".
        model (str, optional): Model name to use. Default is "gpt-3.5-turbo-0613".
        api_key (str, optional): API key for authentication. If None, uses environment variable.
        verbose (bool, optional): If True, enables detailed logging. Default is True.
        as_str (bool, optional): If True, returns only the model's message as string, otherwise returns the entire response object. Default is False.
        **kwargs: Additional parameters to be passed to the `chat_strings` function.

    Returns:
        Union[Coroutine, openai.ChatCompletion, str]: Coroutine object representing the asynchronous execution of the API requests if save_filepath is provided. Otherwise, returns the chat response object, or a string if as_str is True.
    """
    if save_filepath == "chat_temp.txt":
        with open(save_filepath,'w') as f:
            f = ''
    if not isinstance(system_messages, list): system_messages = [system_messages]
    if not isinstance(prompts, list): # If no save_filepath we assume we can skip saving
        if save_filepath == "chat_temp.txt":
            return call(prompts, system_messages[0], model = model, as_str = as_str)
        else:
            prompts = [prompts]
    if len(system_messages) == 1: system_messages = system_messages * len(prompts)
    request_strings = chat_strings(prompts, system_messages, model, **kwargs)
    if api_key is None: api_key = os.environ["OPENAI_API_KEY"]
    job = execute_api_requests_in_parallel(
        request_strings=request_strings,
        save_filepath=save_filepath,
        request_url="https://api.openai.com/v1/chat/completions",
        api_key = api_key,
        logging_level=logging.INFO if verbose else logging.ERROR
    )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In a running event loop (Jupyter Notebooks, IPython), use create_task
            task = asyncio.create_task(job)
        else:
            # Outside notebooks, use run_until_complete
            loop.run_until_complete(job)
    except Exception as E:
        asyncio.run(job)

    return File(save_filepath)[:]


def get_embedding(texts, save_filepath, api_key=None):
    """
    Retrieves embeddings for the given texts by calling the OpenAI API. Processes
    the requests asynchronously and in parallel.

    Parameters:
        texts (list): List of texts for which embeddings are to be retrieved.
        save_filepath (str): File path to save the results.
        api_key (str, optional): API key for authentication. If None, uses environment variable.

    Returns:
        dict: The loaded embeddings from the saved results file.
    """
    # Request strings for jobs (I assume you have a list of jobs you want to process)
    request_strings = [json.dumps(job, ensure_ascii=False) for job in jobs]
    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]

    # Execute API requests in parallel and save results to a file
    job = execute_api_requests_in_parallel(
        request_strings=request_strings,
        save_filepath=save_filepath,
        request_url="https://api.openai.com/v1/embeddings",
        api_key=api_key,
        max_requests_per_minute=3_000 * 0.5,
        max_tokens_per_minute=250_000 * 0.5,
        token_encoding_name="cl100k_base",
        max_attempts=3,
        logging_level=logging.INFO,
    )
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In a running event loop (Jupyter Notebooks, IPython), use create_task
            task = asyncio.create_task(job)
        else:
            # Outside notebooks, use run_until_complete
            loop.run_until_complete(job)
    except:
        asyncio.run(job)

    return File(save_filepath)


class File:
    def __init__(self, path):
        self.path = path
        self.values = None

    def load(self):
        if self.values is None:
            try:
                with open(self.path, 'r') as file:
                    self.values = [eval(line.replace(' null',' None')) for line in file.readlines()]
            except FileNotFoundError:
                print(f"File not found: {self.path}")
            except Exception as e:
                print(f"Error loading file: {e}")

    @property
    def prompts(self):
        if self.values is None:
            self.load()
        return [entry[0] for entry in self.values] if self.values else []

    @property
    def completions(self):
        if self.values is None:
            self.load()
        return [entry[1] for entry in self.values] if self.values else []

    @property
    def _prompts(self):
        if self.values is None:
            self.load()
        return [entry["messages"][1]["content"] for entry in self.prompts] if self.prompts else []

    @property
    def _completions(self):
        if self.values is None:
            self.load()
        return [entry["choices"][0]["message"]["content"] for entry in self.completions] if self.completions else []

    def __getitem__(self, index):
        if self.values is None:
            self.load()
        return self.values[index] if self.values else None

# auth()
# # f = chat('a','a','test')
# # Example usage
# prompts = ["""You are taking part in a study, where you've been asked to solve the following puzzle:

# Peter is looking at Linda
# Linda is looking at Henry
# Peter is married
# Henry is not married
# Is someone who is married looking at someone who is not married?
# You have answered: We can't tell
# Your explanation: Based on the given information, we know that Peter is married and Henry is not married. However, we do not have any information about Linda's marital status. 

# A Message to you:

# Hi,
# Thank you for your well-thought-out response. I appreciate the clarity and critical thinking you’ve brought to the discussion. However, there seems to be a small misunderstanding which I'd like to address.

# Let’s reexamine the question: “Is someone who is married looking at someone who is not married?” The question does not require that it be the same married person in every scenario; rather, it asks if, in the given situation, there exists at least one instance of a married person looking at an unmarried person.

# Now let's revisit the scenarios I presented:

# 1. If Linda is married, then she is looking at Henry, who is not married. In this scenario, Linda (married) is looking at Henry (not married), fulfilling the condition.
   
# 2. If Linda is not married, then we have Peter (who is married) looking at Linda (not married). In this scenario, Peter (married) is looking at Linda (not married), again fulfilling the condition.

# Note that the question doesn't ask if the same married person is looking at an unmarried person in all cases. It asks whether, in the information provided, there is at least one instance of this happening. Since we have established that in either scenario, regardless of Linda’s marital status, there is at least one instance of a married person looking at an unmarried person, the answer must be “yes”.

# Your concern seems to stem from the idea that the “someone who is married” must remain constant across both scenarios. However, this is not a requirement of the question. It merely asks if such a situation exists within the information provided, not whether it’s consistently the same individual who is married.

# I hope this clears up the confusion and helps in understanding why the answer is indeed "yes".

# Best regards,

# """]
# system_messages = ["You are a participant in a psychology study, your behaviour has been encoded in function inputs"]


def revise_your_answer(thought: str, new_answer: str, new_explanation: str):
    """Decide to revise your answer based on the message response.

    Parameters:
        thought (string): The internal thought process behind the decision.
        new_answer (string): The revised answer.
        new_explanation (string): The explanation for the revised answer.
    """
    pass



# json_schema = generate_schema_from_function(function)

# function_call = "auto"

# file = chat(prompts, system_messages, "test_agent", functions=[json_schema], function_call=function_call)
