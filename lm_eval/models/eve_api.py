import asyncio
import copy
import json
import logging
import os
import requests
import threading
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from .api_models import JsonChatStr, TemplateAPI


eval_logger = logging.getLogger(__name__)


@register_model("eve-api")
class EveAPI(TemplateAPI):
    """
    Eve API model for lm_eval.

    This model interfaces with the Eve API which provides RAG-enhanced
    Earth Observation (EO) responses.

    Example usage:
        lm_eval --model eve-api \
                --model_args "email=user@example.com,password=mypass,base_url=https://api.eve.com,public_collections=['qwen-512-filtered','wikipedia-512'],k=5,threshold=0.5" \
                --tasks mmlu
    """

    def __init__(
        self,
        email: str = None,
        password: str = None,
        base_url: str = None,
        public_collections: Optional[List[str]] = None,
        k: int = 5,
        threshold: float = 0.5,
        tokenizer_backend: str = None,
        **kwargs,
    ):
        """
        Initialize the Eve API model.

        Args:
            email: Email for Eve API authentication
            password: Password for Eve API authentication
            base_url: Base URL for the Eve API (e.g., 'https://api.eve.com')
            public_collections: List of collections to search in RAG
            k: Number of documents to retrieve in RAG
            threshold: Similarity threshold for RAG retrieval
            tokenizer_backend: Tokenizer backend to use
            **kwargs: Additional arguments passed to TemplateAPI
        """
        # Store authentication credentials before calling super().__init__
        self.email = email
        self.password = password
        self._base_url = base_url

        # Validate required credentials
        if not self.email or not self.password or not self._base_url:
            raise ValueError(
                "Eve API requires 'email', 'password', and 'base_url' to be set. "
                "Pass them via --model_args 'email=...,password=...,base_url=...'"
            )

        # Construct the generate endpoint URL
        generate_url = f"{self._base_url}/generate"

        super().__init__(
            base_url=generate_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=False,  # Eve API takes text queries
            **kwargs,
        )

        # Store RAG parameters
        # Handle case where public_collections comes as a string (from command-line parsing)
        eval_logger.info(
            f"[EVE_API INIT] Received public_collections type: {type(public_collections)}, value: {repr(public_collections)}"
        )

        if public_collections is None:
            eval_logger.info(
                "[EVE_API INIT] public_collections is None, using defaults"
            )
            self.public_collections = ["qwen-512-filtered", "wikipedia-512"]
        elif isinstance(public_collections, str):
            eval_logger.info(
                f"[EVE_API INIT] public_collections is a string, attempting to parse"
            )
            # Check if it's a pipe-separated list (our custom format to avoid comma conflicts)
            if "|" in public_collections:
                self.public_collections = [
                    c.strip() for c in public_collections.split("|") if c.strip()
                ]
                eval_logger.info(
                    f"[EVE_API INIT] Split pipe-separated string into list: {self.public_collections}"
                )
            else:
                # Single collection name
                eval_logger.info(f"[EVE_API INIT] Using as single collection")
                self.public_collections = [public_collections]
        elif isinstance(public_collections, list):
            eval_logger.info(
                f"[EVE_API INIT] public_collections is already a list, using as-is: {public_collections}"
            )
            self.public_collections = public_collections
        else:
            eval_logger.info(
                f"[EVE_API INIT] public_collections is a {type(public_collections)}, using as-is"
            )
            self.public_collections = public_collections

        self.k = int(k)
        self.threshold = float(threshold)

        # Store authentication token (will be lazy-loaded)
        self._token = None
        self._token_lock = None  # Will be set to asyncio.Lock() when needed

        # Thread-safe storage for API responses (for logging full responses)
        self._response_storage = {}
        self._storage_lock = threading.Lock()

        eval_logger.info(f"Initialized Eve API model with base_url={self.base_url}")
        eval_logger.info(
            f"RAG parameters: collections={self.public_collections}, k={self.k}, threshold={self.threshold}"
        )

    def _authenticate(self, force_refresh: bool = False) -> str:
        """
        Authenticate with the Eve API and return the access token.

        Args:
            force_refresh: If True, force a new authentication even if a token exists

        Returns:
            str: The access token
        """
        if not force_refresh and self._token is not None:
            return self._token

        try:
            eval_logger.info(f"Authenticating with Eve API at {self._base_url}/login")
            response = requests.post(
                f"{self._base_url}/login",
                json={"email": self.email, "password": self.password},
                timeout=self.timeout,
            )
            response.raise_for_status()
            token = response.json()["access_token"]
            self._token = token
            eval_logger.info("Successfully authenticated with Eve API")
            return token
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Failed to authenticate with Eve API: {e}")
            raise ValueError(f"Eve API authentication failed: {e}")

    def refresh_token(self):
        """Invalidate the current token and get a new one."""
        eval_logger.info("Refreshing Eve API token")
        self._token = None
        return self._authenticate(force_refresh=True)

    @property
    def access_token(self) -> str:
        """Get or refresh the access token."""
        if self._token is None:
            self._token = self._authenticate()
        return self._token

    def _create_payload(
        self,
        messages: Union[List[Dict], str],
        generate=True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        """
        Create the payload for the Eve API request.

        Args:
            messages: The input query (as string or chat messages)
            generate: Whether this is a generation request
            gen_kwargs: Generation parameters (not used for Eve API)
            seed: Random seed (not used for Eve API)
            eos: End of sequence token (not used for Eve API)
            **kwargs: Additional arguments

        Returns:
            dict: The API payload
        """
        # Extract query from messages
        if isinstance(messages, str):
            query = messages
        elif isinstance(messages, list) and len(messages) > 0:

            # Apply chat format: concatenate all user/system messages into a single query string
            query = "\n".join(
                [msg.get("content", "") for msg in messages if "content" in msg]
            )
        else:
            query = str(messages)

        # Create Eve API payload
        eval_logger.info(
            f"[EVE_API PAYLOAD] self.public_collections type: {type(self.public_collections)}, value: {repr(self.public_collections)}"
        )

        payload = {
            "query": query,
            "public_collections": self.public_collections,
            "k": self.k,
            "score_threshold": self.threshold,
            "llm_type": "main",
        }

        eval_logger.info(
            f"[EVE_API] Created payload with public_collections: {self.public_collections}"
        )
        eval_logger.info(f"[EVE_API] Full payload: {payload}")
        return payload

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """
        Parse the Eve API response to extract generated text.

        Args:
            outputs: The API response(s)
            **kwargs: Additional arguments

        Returns:
            List[str]: List of generated texts
        """
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            try:
                # Extract the 'answer' field from the Eve API response
                answer = out.get("answer", "")
                res.append(answer)
            except Exception as e:
                eval_logger.warning(f"Could not parse Eve API response: {e}")
                res.append("")

        return res

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[tuple]:
        """
        Parse logprobs from Eve API response.

        Note: Eve API does not support logprobs, so this method should never be called.
        It's only implemented to satisfy the abstract method requirement.
        """
        raise NotImplementedError(
            "Logprobs parsing is not supported for Eve API. "
            "Eve API is a generation-only API."
        )

    def loglikelihood(self, requests, **kwargs):
        """
        Eve API does not support loglikelihood computation.

        This is typical for generation-only APIs.
        """
        raise NotImplementedError(
            "Loglikelihood is not supported for Eve API. "
            "Eve API is a generation-only API and cannot compute prompt logprobs. "
            "You can only use it with generation tasks, not multiple-choice tasks."
        )

    @property
    def api_key(self):
        """
        Return the access token for API requests.

        This property is used by the TemplateAPI base class to add
        authentication headers to requests.
        """
        return self.access_token

    @property
    def header(self) -> dict:
        """
        Override header property to make it non-cached.

        This allows the token to be refreshed when needed.
        """
        return {"Authorization": f"Bearer {self.access_token}"}

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        """
        Override synchronous model_call to store full API responses.

        This is used when num_concurrent <= 1 (synchronous execution).
        """
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)

        # Create payload
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )

        eval_logger.info(f"[EVE_API SYNC] Sending POST to {self.base_url}")
        eval_logger.info(f"[EVE_API SYNC] Payload: {payload}")

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.header,
                verify=self.verify_certificate,
                timeout=self.timeout,
            )

            if not response.ok:
                # Handle 401 - token expired
                if response.status_code == 401:
                    eval_logger.warning(
                        "Received 401 Unauthorized. Refreshing token and retrying..."
                    )
                    self.refresh_token()

                    # Retry with new token
                    response = requests.post(
                        self.base_url,
                        json=payload,
                        headers=self.header,  # Will use new token
                        verify=self.verify_certificate,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                else:
                    eval_logger.warning(
                        f"API request failed with error message: {response.text}. Retrying..."
                    )
                    response.raise_for_status()

            outputs = response.json()

            # Store the full API response for later logging
            # Use the raw messages as the storage key
            # Extract the string from messages
            if messages and len(messages) > 0:
                # messages could be a list of JsonChatStr or strings
                message = messages[0]
                if hasattr(message, 'prompt'):
                    storage_key = message.prompt
                else:
                    storage_key = str(message)

                with self._storage_lock:
                    # Store both request payload and full response
                    self._response_storage[storage_key] = {
                        "request": copy.deepcopy(payload),
                        "response": copy.deepcopy(outputs) if isinstance(outputs, dict) else outputs,
                    }
                eval_logger.info(
                    f"[EVE_API SYNC] Stored full response for storage_key type={type(storage_key).__name__}, "
                    f"first 100 chars: {str(storage_key)[:100]}, total stored: {len(self._response_storage)}"
                )

            return outputs

        except requests.exceptions.RequestException as e:
            eval_logger.error(f"[EVE_API SYNC] Request failed: {e}")
            raise

    async def amodel_call(
        self,
        session,
        sem,
        messages,
        *,
        generate: bool = True,
        cache_keys: list = None,
        ctxlens=None,
        gen_kwargs=None,
        **kwargs,
    ):
        """
        Override amodel_call to handle 401 errors, refresh token, and store full API responses.
        """
        from aiohttp import ClientResponseError

        # Initialize token lock if needed
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()

        # Create payload
        gen_kwargs_copy = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs_copy,
            seed=self._seed,
            **kwargs,
        )

        eval_logger.info(f"[EVE_API] Sending POST to {self.base_url}")
        eval_logger.info(f"[EVE_API] Payload being sent: {payload}")

        cache_method = "generate_until" if generate else "loglikelihood"
        acquired = await sem.acquire()

        try:
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    # Handle 401 - token expired
                    if response.status == 401:
                        error_text = await response.text()
                        eval_logger.warning(
                            f"Received 401 Unauthorized. Refreshing token and retrying..."
                        )

                        # Save current token before acquiring lock
                        old_token = self._token
                        async with self._token_lock:
                            # Check if another coroutine already refreshed the token
                            if self._token == old_token or self._token is None:
                                # Refresh the token (synchronous call)
                                self.refresh_token()
                                eval_logger.info("Token refreshed successfully")

                        # Retry with new token
                        eval_logger.info("Retrying request with new token")
                        async with session.post(
                            self.base_url,
                            json=payload,
                            headers=self.header,  # Will use new token
                        ) as retry_response:
                            retry_response.raise_for_status()
                            outputs = await retry_response.json()
                    else:
                        # Other errors
                        error_text = await response.text()
                        eval_logger.warning(
                            f"API request failed! Status code: {response.status}, "
                            f"Response text: {error_text}. Retrying..."
                        )
                        response.raise_for_status()
                else:
                    # Success - get the response
                    outputs = await response.json()

            # Parse the response to get answers
            answers = (
                self.parse_generations(outputs=outputs)
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )

            # Store the full API response for later logging
            # Use the context (first element of cache_key) as the storage key
            # Note: cache_keys are tuples of (context, gen_kwargs), but gen_kwargs is a dict
            # and can't be used as a dict key. Context alone is sufficient for uniqueness.
            if cache_keys:
                # Store response for each cache key
                # Note: For Eve API, we typically have one message per request
                # but we handle multiple just in case
                for idx, (res, cache_key) in enumerate(zip(answers, cache_keys)):
                    try:
                        # Extract context from cache_key tuple (context, gen_kwargs)
                        if isinstance(cache_key, tuple) and len(cache_key) >= 1:
                            storage_key = cache_key[0]
                        else:
                            storage_key = cache_key

                        # Convert JsonChatStr to string for storage if needed
                        # JsonChatStr is a NamedTuple with a 'prompt' field
                        if hasattr(storage_key, 'prompt'):
                            storage_key = storage_key.prompt

                        with self._storage_lock:
                            # Store both request payload and full response
                            # Use storage_key as key (it's unique per request)
                            self._response_storage[storage_key] = {
                                "request": copy.deepcopy(payload),
                                "response": (
                                    copy.deepcopy(outputs) if isinstance(outputs, dict) else outputs
                                ),
                            }
                        eval_logger.info(
                            f"[EVE_API] Stored full response for storage_key type={type(storage_key).__name__}, "
                            f"first 100 chars: {str(storage_key)[:100]}, total stored: {len(self._response_storage)}"
                        )
                    except Exception as e:
                        eval_logger.error(f"[EVE_API] Error storing response for cache_key {idx}: {e}")
                        import traceback
                        eval_logger.error(traceback.format_exc())

                    # Add to cache
                    self.cache_hook.add_partial(cache_method, cache_key, res)
            else:
                # Fallback: store by query if no cache_keys (shouldn't happen in practice)
                query = payload.get("query", "")
                with self._storage_lock:
                    self._response_storage[query] = {
                        "request": copy.deepcopy(payload),
                        "response": (
                            copy.deepcopy(outputs) if isinstance(outputs, dict) else outputs
                        ),
                    }
                eval_logger.warning(
                    f"[EVE_API] No cache_keys provided, storing by query (first 100 chars): {query[:100]}"
                )

            return answers

        except BaseException as e:
            # If outputs is not defined, define it here for logging
            if "outputs" not in locals():
                outputs = None
            eval_logger.error(f"Exception: {repr(e)}, {outputs}, retrying.")
            raise e
        finally:
            if acquired:
                sem.release()

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """
        Override generate_until to attach full API responses to instances for logging.

        This ensures that the full Eve API response (including retrieved documents,
        scores, etc.) is saved to the samples JSONL file and uploaded to wandb.
        """
        # Call parent method to get results
        results = super().generate_until(requests, disable_tqdm)

        # Attach the full API response to each instance for sample logging
        eval_logger.info(
            f"[EVE_API] Attaching full responses to {len(requests)} instances"
        )

        for instance, result in zip(requests, results):
            # Get the original arguments (context/query, gen_kwargs)
            if hasattr(instance, "args") and len(instance.args) >= 2:
                context, gen_kwargs = instance.args[0], instance.args[1]

                try:
                    # Use context as the storage key (same as in amodel_call)
                    # Convert JsonChatStr to string if needed
                    lookup_key = context
                    if hasattr(lookup_key, 'prompt'):
                        lookup_key = lookup_key.prompt

                    with self._storage_lock:
                        api_data = self._response_storage.get(lookup_key)
                        storage_info = f"lookup_key type: {type(lookup_key).__name__}, storage keys count: {len(self._response_storage)}"

                    if api_data is not None:
                        # Attach the full API data (request + response) to the instance
                        # This will be included in samples output
                        instance.eve_api_request = api_data["request"]
                        instance.eve_api_response = api_data["response"]

                        eval_logger.debug(
                            f"[EVE_API] Attached full response to instance {instance.idx}"
                        )
                    else:
                        # Try to extract query for debugging
                        try:
                            if isinstance(lookup_key, str):
                                try:
                                    messages = json.loads(lookup_key)
                                    if isinstance(messages, list) and len(messages) > 0:
                                        query = "\n".join(
                                            [
                                                msg.get("content", "")
                                                for msg in messages
                                                if "content" in msg
                                            ]
                                        )
                                    else:
                                        query = lookup_key
                                except (json.JSONDecodeError, TypeError):
                                    query = lookup_key
                            else:
                                query = str(lookup_key)
                        except Exception:
                            query = "unknown"

                        eval_logger.warning(
                            f"[EVE_API] Could not find stored response for instance {instance.idx}, "
                            f"lookup_key (first 50 chars): {str(lookup_key)[:50]}, {storage_info}"
                        )
                        instance.eve_api_request = {"query": query[:100] + "..." if len(query) > 100 else query}
                        instance.eve_api_response = {
                            "error": "Response not found in storage",
                            "debug_info": storage_info
                        }

                except Exception as e:
                    eval_logger.warning(
                        f"[EVE_API] Could not attach response to instance: {e}"
                    )
                    import traceback
                    eval_logger.error(traceback.format_exc())
                    # Set a minimal response on error
                    instance.eve_api_request = {"error": str(e)}
                    instance.eve_api_response = {"error": str(e)}

        eval_logger.info(f"[EVE_API] Successfully attached responses to all instances")
        return results
