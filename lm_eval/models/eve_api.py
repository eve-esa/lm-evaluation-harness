import logging
import os
import requests
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.registry import register_model
from .api_models import TemplateAPI


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
        Override amodel_call to handle 401 errors and refresh token.
        """
        import asyncio
        from aiohttp import ClientResponseError

        # Initialize token lock if needed
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()

        try:
            # Try the request with current token
            return await super().amodel_call(
                session=session,
                sem=sem,
                messages=messages,
                generate=generate,
                cache_keys=cache_keys,
                ctxlens=ctxlens,
                gen_kwargs=gen_kwargs,
                **kwargs,
            )
        except ClientResponseError as e:
            # If we get a 401, refresh the token and retry once
            if e.status == 401:
                eval_logger.warning(
                    "Received 401 Unauthorized. Refreshing token and retrying..."
                )
                async with self._token_lock:
                    # Check if another coroutine already refreshed the token
                    old_token = self._token
                    if old_token == self._token or self._token is None:
                        # Refresh the token
                        self.refresh_token()
                        eval_logger.info("Token refreshed successfully")

                # Retry the request with new token
                eval_logger.info("Retrying request with new token")
                return await super().amodel_call(
                    session=session,
                    sem=sem,
                    messages=messages,
                    generate=generate,
                    cache_keys=cache_keys,
                    ctxlens=ctxlens,
                    gen_kwargs=gen_kwargs,
                    **kwargs,
                )
            else:
                # Re-raise other errors
                raise
