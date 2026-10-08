import re

SAFE_NAME_PATTERN = r"^[A-Za-z0-9_\-]+$"
_SAFE_NAME_RE = re.compile(SAFE_NAME_PATTERN)


def validate_slug(value: str, field_name: str = "value") -> str:
    # fullmatch: re.match+$ だと末尾改行を許容してしまう(Pythonのreの既知の挙動)ため、
    # 文字列全体との完全一致をfullmatchで保証する。
    if not _SAFE_NAME_RE.fullmatch(value):
        raise ValueError(
            f"{field_name}は英数字・アンダースコア・ハイフンのみ使用できます: {value!r}"
        )
    return value
