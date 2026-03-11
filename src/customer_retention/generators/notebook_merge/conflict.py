import re

_START_RE = re.compile(r"^# <<<< \S+")
_END_RE = re.compile(r"^# >>>> \S+")
_SEP = "# ===="


def wrap_conflict(
    theirs_source: str, ours_source: str,
    theirs_label: str = "theirs", ours_label: str = "ours",
) -> str:
    t = theirs_source.rstrip("\n")
    o = ours_source.rstrip("\n")
    return f"# <<<< {theirs_label}\n{t}\n{_SEP}\n{o}\n# >>>> {ours_label}"


def has_conflict_markers(source: str) -> bool:
    has_start = has_sep = has_end = False
    for line in source.split("\n"):
        if _START_RE.match(line):
            has_start = True
        elif line.strip() == _SEP:
            has_sep = True
        elif _END_RE.match(line):
            has_end = True
    return has_start and has_sep and has_end
