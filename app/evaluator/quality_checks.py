import json


def has_content(output: str) -> bool:
    return bool(output and output.strip())


def is_valid_json(output: str) -> bool:
    try:
        json.loads(output)
        return True
    except Exception:
        return False


def has_code_block(output: str) -> bool:
    return "```" in output
