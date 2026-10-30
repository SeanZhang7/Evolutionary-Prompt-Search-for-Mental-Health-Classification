"""Qwen generation settings and a single request/response operation.

This module has no CLI, templates, population management or candidate generation
loop. The GA runner supplies the request and owns parsing and retry decisions.
"""

import json
import math
import time
from urllib.error import HTTPError, URLError

if __package__:
    from . import depsign_evaluator as evaluator
else:
    import depsign_evaluator as evaluator


def load_settings(config):
    model = config["model"]
    if model["checkpoint"] != "Qwen/Qwen3.5-9B" or not model["same_checkpoint_for_both_roles"]:
        raise ValueError("This runner requires the same Qwen3.5-9B checkpoint for both roles")
    parameters = dict(model["generation"])
    if parameters.pop("decoding") != "sampling":
        raise ValueError("Generation must use sampling")
    if parameters.pop("seed_policy") != "sha256_search_seed_condition_slot_attempt_operator":
        raise ValueError("Unsupported generation seed policy")
    expected = set(evaluator.PARAMETERS) - {"seed"}
    if set(parameters) != expected:
        raise ValueError("Unexpected generation parameters")
    temperature = parameters["temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Generation temperature must be a finite number greater than 0")
    chat_kwargs = parameters["chat_template_kwargs"]
    if (not isinstance(chat_kwargs, dict) or set(chat_kwargs) != {"enable_thinking"}
            or type(chat_kwargs["enable_thinking"]) is not bool):
        raise ValueError("Generation chat_template_kwargs must contain a boolean enable_thinking")
    if type(parameters["max_tokens"]) is not int or parameters["max_tokens"] < 1:
        raise ValueError("Generation max_tokens must be a positive integer (reasoning plus final output)")
    search = config["search"]
    if search["max_instruction_tokens"] != evaluator.MAX_INSTRUCTION_TOKENS:
        raise ValueError("Generator and evaluator instruction limits differ")
    attempts = search["max_proposal_attempts_per_slot"]
    retries = search["max_transport_retries_per_request"]
    if type(attempts) is not int or attempts < 1 or type(retries) is not int or retries < 0:
        raise ValueError("Invalid attempt or transport retry limits")
    return parameters, attempts, retries


def generate_once(request):
    """Send one supplied request; preserve the response without retrying or parsing prompts."""
    result = {
        "raw_http_response": None, "raw_output": None, "request_error": None,
        "reasoning": None, "finish_reason": None, "usage": None, "retryable": False,
    }
    started = time.monotonic()
    try:
        result["raw_http_response"] = evaluator.http_call("/v1/chat/completions", request)
        response = json.loads(result["raw_http_response"])
        choice = response["choices"][0]
        message = choice["message"]
        result.update(raw_output=message.get("content"),
                      reasoning=message.get("reasoning") or message.get("reasoning_content"),
                      finish_reason=choice.get("finish_reason"), usage=response.get("usage"))
    except HTTPError as exc:
        result.update(raw_http_response=exc.read().decode("utf-8", errors="replace"),
                      request_error=f"HTTP {exc.code}", retryable=exc.code == 429 or 500 <= exc.code < 600)
    except (URLError, TimeoutError, OSError) as exc:
        result.update(request_error=f"{type(exc).__name__}: {exc}", retryable=True)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        result["request_error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.monotonic() - started
    return result
