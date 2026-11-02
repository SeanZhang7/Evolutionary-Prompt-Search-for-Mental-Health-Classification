"""Run-local, per-parent critique requests for the v5 GA operator."""

import json
import math
import re

if __package__:
    from . import depsign_evaluator as evaluator, qwen_client
else:
    import depsign_evaluator as evaluator
    import qwen_client


def load_settings(config, prompt_bytes):
    prompt = json.loads(prompt_bytes)
    if (not isinstance(prompt, dict) or set(prompt) != {"name", "instruction"}
            or any(not isinstance(v, str) or not v.strip() for v in prompt.values())):
        raise ValueError("Critique prompt requires nonempty name and instruction strings")
    settings = config["model"]["critique"]
    if set(settings) != {"temperature", "top_p", "top_k", "max_tokens", "chat_template_kwargs"}:
        raise ValueError("Unexpected critique settings")
    for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1)):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low < value <= high:
            raise ValueError(f"Invalid critique {key}")
    if type(settings["top_k"]) is not int or (settings["top_k"] != -1 and settings["top_k"] < 1):
        raise ValueError("Invalid critique top_k")
    if type(settings["max_tokens"]) is not int or settings["max_tokens"] < 1:
        raise ValueError("Critique max_tokens must be a positive integer")
    kwargs = settings["chat_template_kwargs"]
    if not isinstance(kwargs, dict) or set(kwargs) != {"enable_thinking"} or kwargs["enable_thinking"] is not False:
        raise ValueError("Critique thinking must be disabled")
    parameters = {k: v for k, v in evaluator.PARAMETERS.items() if k != "seed"}
    return prompt, {**parameters, **settings}


def build_request(task, parent, prompt, parameters, tokenizer, *, allow_unscored=False):
    # Criticism needs label definitions, not demonstration posts or the classifier wrapper.
    definitions = re.split(r"\r?\n(?:Few-shot examples|Instruction)[ \t]*\r?\n", task, maxsplit=1)[0]
    if "performance" not in parent and not allow_unscored:
        raise ValueError(f"Missing measured performance for parent {parent['id']}")
    record = {"instruction": parent["instruction"],
              "confusion_matrix": parent.get("performance", {}).get("confusion_matrix")}
    if allow_unscored and record["confusion_matrix"] is None:
        record["feedback_status"] = "not_evaluated_preparation_placeholder_only"
    content = (prompt["instruction"] + "\n\nTask definitions (JSON string):\n"
               + json.dumps(definitions.strip(), ensure_ascii=False)
               + "\n\nParent record (JSON):\n" + json.dumps(record, ensure_ascii=False))
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    if tokens + parameters["max_tokens"] > evaluator.CONTEXT_TOKENS:
        raise ValueError(f"Critique prompt plus output exceeds context budget: {tokens} + "
                         f"{parameters['max_tokens']} > {evaluator.CONTEXT_TOKENS}")
    return {"model": evaluator.MODEL, "messages": messages, **parameters}


class ParentCritic:
    def __init__(self, task, prompt, parameters, tokenizer, seed, report):
        self.task, self.prompt, self.parameters = task, prompt, parameters
        self.tokenizer, self.seed, self.report = tokenizer, seed, report
        self.cache = {}

    def critique(self, parent):
        identity = {"parent_id": parent["id"], "instruction": parent["instruction"],
                    "confusion_matrix": parent["performance"]["confusion_matrix"]}
        key = evaluator.digest(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode())
        if key in self.cache:
            self.report["cache_hits"] += 1
            return self.cache[key]
        request = build_request(self.task, parent, self.prompt, self.parameters, self.tokenizer)
        entry = {**identity, "input_sha256": key, "critique": None, "attempts": []}
        self.report["parents"].append(entry)
        for attempt in range(1, 4):
            seed_key = json.dumps([self.seed, "parent_critique", key, attempt]).encode()
            payload = {**request, "seed": int(evaluator.digest(seed_key)[:8], 16) % (2 ** 31)}
            for retry in range(2):
                event = {"attempt": attempt, "transport_retry": retry, "seed": payload["seed"]}
                entry["attempts"].append(event)
                self.report["requests"] += 1
                self.report["transport_retries"] += retry
                try:
                    result = qwen_client.generate_once(payload)
                except (Exception, KeyboardInterrupt) as exc:
                    event.update(outcome="interrupted" if isinstance(exc, KeyboardInterrupt) else "request_error",
                                 error=f"{type(exc).__name__}: {exc}")
                    raise
                usage = result.get("usage")
                event["usage"] = usage
                if isinstance(usage, dict):
                    self.report["responses_with_usage"] += 1
                    for field in self.report["reported_usage"]:
                        self.report["reported_usage"][field] += usage.get(field, 0) or 0
                if result["request_error"]:
                    event.update(outcome="request_error", error=result["request_error"])
                    if result["retryable"] and retry == 0:
                        continue
                    raise RuntimeError(f"Critique request failed for {parent['id']}: {result['request_error']}")
                text = result["raw_output"]
                reason = None
                if result["finish_reason"] != "stop":
                    reason = f"Expected finish_reason=stop; got {result['finish_reason']!r}"
                elif result["reasoning"]:
                    reason = "Nonempty reasoning despite critique thinking disabled"
                elif not isinstance(text, str) or not text.strip():
                    reason = "Empty critique"
                elif "<think>" in text or "</think>" in text:
                    reason = "Reasoning tags in critique content"
                event.update(outcome="invalid" if reason else "accepted", rejection_reason=reason,
                             finish_reason=result["finish_reason"])
                print(f"Critique {parent['id']}, attempt {attempt}, retry {retry}: {event['outcome']}"
                      + (f" — {reason}" if reason else ""), flush=True)
                if reason is None:
                    entry["critique"] = text.strip()
                    self.cache[key] = entry["critique"]
                    return entry["critique"]
                break
        raise RuntimeError(f"Critique for {parent['id']} exhausted 3 proposals; last rejection: {reason}")
