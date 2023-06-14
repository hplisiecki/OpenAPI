import asyncio
import json
from typing import List
import logging
from api_functions import (
    process_api_requests_from_file,
    api_endpoint_from_url,
    num_tokens_consumed_from_request,
    APIRequest,
    StatusTracker
)


async def execute_api_requests_in_parallel(
        request_strings: List[str],
        save_filepath: str,
        request_url: str,
        api_key: str,
        max_requests_per_minute: float =3_000 * 0.2,
        max_tokens_per_minute: float =250_000 * 0.2,
        token_encoding_name: str = "cl100k_base",
        max_attempts: int = 3,
        logging_level: int = logging.INFO,
):
    # create list of request JSON objects
    requests = [json.loads(request_string) for request_string in request_strings]

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize asyncio event loop
    loop = asyncio.get_event_loop()

    # create tasks for each request
    tasks = [
        loop.create_task(
            APIRequest(
                task_id=task_id,
                request_json=request,
                token_consumption=num_tokens_consumed_from_request(request, api_endpoint, token_encoding_name),
                attempts_left=max_attempts,
            ).call_api(
                request_url=request_url,
                request_header=request_header,
                retry_queue=asyncio.Queue(),
                save_filepath=save_filepath,
                status_tracker=StatusTracker(),
            )
        )
        for task_id, request in enumerate(requests, start=1)
    ]

    # run tasks in parallel
    await asyncio.gather(*tasks)
