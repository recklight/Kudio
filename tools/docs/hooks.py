# -*- coding: utf-8 -*-
"""Build-time pieces of the documentation site, wired in from mkdocs.yml.

The API overview page lists every name ``import kudio`` gives you, grouped
the way the reference pages are. It is generated here from ``kudio.__all__``
rather than written by hand, so a name added to the library shows up on the
site without anyone remembering to add it. The grouping is read from the
``::: module`` lines of the reference pages themselves, so moving a module to
another page moves its names too.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kudio

#: Where in ``docs/reference/index.md`` the generated tables go.
MARKER = "<!-- kudio:top-level-index -->"
_IDENT = re.compile(r"^:::\s+(kudio[\w.]*)\s*$", re.M)
_TITLE = re.compile(r"^#\s+(.+)$", re.M)
_ROLE = re.compile(r":(?:func|class|meth|attr|data|mod|exc|obj):`~?([^`]+)`")
_DEPRECATED = "Deprecated alias for"


def _summary(obj) -> str:
    """The first sentence of a docstring, fit for one table cell."""
    doc = inspect.getdoc(obj) or ""
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    first = _ROLE.sub(r"`\1`", first)
    end = re.search(r"(?<=[.;])\s", first)
    if end:
        first = first[:end.start()]
    return first.replace("|", "\\|")


def _defined_in(name: str, obj) -> Optional[str]:
    """The module whose source assigns *name* at top level, for constants,
    which carry no ``__module__`` of their own."""
    for modname, module in sorted(sys.modules.items()):
        if not modname.startswith("kudio.") or module is None:
            continue
        if vars(module).get(name) is not obj:
            continue
        try:
            source = inspect.getsource(module)
        except (OSError, TypeError):
            continue
        if re.search(rf"^{re.escape(name)}\s*[:=]", source, re.M):
            return modname
    return None


def _documented_constant(module: str, name: str) -> bool:
    """A constant gets its own anchor only when a ``#:`` comment documents it
    (see tools/docs/griffe_markdown.py)."""
    source = inspect.getsource(sys.modules[module]).split("\n")
    for i, line in enumerate(source):
        if re.match(rf"^{re.escape(name)}\s*[:=]", line):
            return i > 0 and source[i - 1].strip().startswith("#:")
    return False


def _reference_pages(config, docs_dir: Path) -> List[Tuple[str, str, List[str]]]:
    """(title, page, modules) for each reference page, in navigation order."""
    pages: List[str] = []

    def collect(items):
        for item in items:
            value = list(item.values())[0] if isinstance(item, dict) else item
            if isinstance(value, list):
                collect(value)
            elif isinstance(value, str) and value.startswith("reference/") \
                    and value != "reference/index.md":
                pages.append(value)

    collect(config["nav"])
    found = []
    for page in pages:
        text = (docs_dir / page).read_text(encoding="utf-8")
        title = _TITLE.search(text)
        found.append((title.group(1) if title else page, page,
                      _IDENT.findall(text)))
    return found


def _overview(config) -> str:
    docs_dir = Path(config["docs_dir"])
    sections = _reference_pages(config, docs_dir)
    home: Dict[str, int] = {}
    for n, (_, _, modules) in enumerate(sections):
        for module in modules:
            home[module] = n

    rows: List[List[str]] = [[] for _ in sections]
    old_names: List[str] = []
    for name in sorted(kudio.__all__, key=str.lower):
        if name.startswith("__"):
            continue
        obj = getattr(kudio, name)
        if (inspect.isfunction(obj) or inspect.isclass(obj)) \
                and (inspect.getdoc(obj) or "").startswith(_DEPRECATED):
            new = obj.__wrapped__
            old_names.append(f"| `{name}` | [`{new.__name__}`]"
                             f"[{new.__module__}.{new.__qualname__}] |")
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            module = obj.__module__
            # an object with no docstring gets no anchor of its own in the
            # reference, so its row points at the module instead
            target = f"{module}.{obj.__qualname__}" if inspect.getdoc(obj)                 else module
            kind = "class" if inspect.isclass(obj) else "function"
        else:
            module = _defined_in(name, obj)
            if module is None:
                continue
            target = f"{module}.{name}" if _documented_constant(module, name) \
                else module
            kind = "constant"
        section = home.get(module)
        if section is None:
            raise ValueError(
                f"kudio.{name} lives in {module}, which no reference page "
                f"documents; add a '::: {module}' line to one of them")
        rows[section].append(
            f"| [`{name}`][{target}] | {kind} | {_summary(obj)} |")

    out = []
    for (title, page, _), lines in zip(sections, rows):
        if not lines:
            continue
        out += [f"## {title}", "",
                f"Documented in full on [{title}]({Path(page).name}).", "",
                "| Name | Kind | What it does |", "|---|---|---|", *lines, ""]
    if old_names:
        out += ["## Old names", "",
                "Still importable, but each call warns and points at the new "
                "name. They will be removed in a future release.", "",
                "| Old name | Use instead |", "|---|---|", *old_names, ""]
    return "\n".join(out)


#: A Chinese character or full-width punctuation mark.
_CJK = re.compile("[　-〿一-鿿＀-￯]")
#: A line that opens a block of its own, so nothing is joined onto it.
_OPENS = re.compile(r"[ \t]*(?:[-*+]\s|\d+\.\s|[|#>:<]|!!!|\?\?\?|===|```|~~~)")
#: A line that ends a block, so the next line is never joined onto it.
_CLOSED = re.compile(r"[ \t]*(?:[|#<]|!!!|\?\?\?|===|```|~~~)")


def _join_cjk_lines(markdown: str) -> str:
    """Join the lines of a Chinese paragraph. A browser renders a line break
    as a space, and next to a Chinese character that space shows; the sources
    stay wrapped at 80 columns for the diffs. Fenced code is left alone."""
    lines = markdown.split("\n")
    out: List[str] = []
    fenced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            out.append(line)
            continue
        prev = out[-1] if out else ""
        if (not fenced and stripped and prev.strip() and not _OPENS.match(line)
                and not _CLOSED.match(prev)
                and (_CJK.match(prev.rstrip()[-1]) or _CJK.match(stripped[0]))):
            out[-1] = prev.rstrip() + stripped
            continue
        out.append(line)
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files, **kwargs) -> str:
    if page.file.src_uri.endswith(".zh-TW.md"):
        markdown = _join_cjk_lines(markdown)
    if MARKER not in markdown:
        return markdown
    return markdown.replace(MARKER, _overview(config))
