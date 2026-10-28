"""
API route definitions and dispatch table.

Status: Clean
"""

import json
from typing import Dict, Any, Tuple, Optional
from .request_handler import RequestHandler
from utils.logging_utils import get_logger

logger = get_logger("routes")


class RouteHandler:
    """
    HTTP route router matching path and method to RequestHandler methods.
    """

    def __init__(self, request_handler: RequestHandler):
        self.handler = request_handler

    def dispatch(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Dispatch request to endpoint handler and return (status_code, response_dict).
        """
        clean_path = path.split("?")[0].rstrip("/")

        logger.debug(f"Dispatching {method} {clean_path}")

        if method == "GET" and clean_path in ("/health", ""):
            return 200, self.handler.handle_health()

        if method == "GET" and clean_path == "/v1/models":
            return 200, self.handler.handle_models()

        if method == "POST" and clean_path in ("/v1/completions", "/completions"):
            request_body = {} if body is None else body
            try:
                response = self.handler.handle_completion(request_body)
            except (ValueError, TypeError) as exc:
                return 400, {"error": str(exc)}
            return 200, response

        return 404, {"error": f"Endpoint not found: {method} {path}"}
