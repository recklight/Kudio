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

from markdown.extensions.toc import slugify

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


def _overview(config, language: str = "en") -> str:
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
            # the name the docstring gives, `kudio.util.red` for one that only
            # moved; and, as below, the module when the new one has no anchor
            shown = _ROLE.search(inspect.getdoc(obj))
            shown = shown.group(1) if shown else new.__name__
            target = f"{new.__module__}.{new.__qualname__}" if inspect.getdoc(new) \
                else new.__module__
            old_names.append(f"| `{name}` | [`{shown}`][{target}] |")
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            module = obj.__module__
            # an object with no docstring gets no anchor of its own in the
            # reference, so its row points at the module instead
            target = f"{module}.{obj.__qualname__}" if inspect.getdoc(obj) \
                else module
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

    words = _WORDS[language]
    names = _nav_names(config, language)
    out = []
    for (title, page, _), lines in zip(sections, rows):
        if not lines:
            continue
        shown = names.get(title, title)
        # the English heading's anchor in both languages, so a link to
        # #audio-files works from either page
        anchor = f" {{#{slugify(title, '-')}}}" if shown != title else ""
        out += [f"## {shown}{anchor}", "",
                words["full"].format(title=shown, page=Path(page).name), "",
                words["head"], "|---|---|---|",
                *(_KIND.sub(lambda m: words[m.group(1)], line) for line in lines), ""]
    if old_names:
        out += [words["old"], "", words["old_text"], "",
                words["old_head"], "|---|---|", *old_names, ""]
    return "\n".join(out)


#: The overview's own words, per language; the summaries in its third column
#: are the docstrings' first sentences and stay in English.
_WORDS = {
    "en": {
        "full": "Documented in full on [{title}]({page}).",
        "head": "| Name | Kind | What it does |",
        "function": "function", "class": "class", "constant": "constant",
        "old": "## Old names",
        "old_text": "Still importable, but each call warns and points at the new "
                    "name. They will be removed in a future release.",
        "old_head": "| Old name | Use instead |",
    },
    "zh-TW": {
        "full": "完整說明在[{title}]({page})。",
        "head": "| 名稱 | 種類 | 用途（取自 docstring） |",
        "function": "函式", "class": "類別", "constant": "常數",
        "old": "## 舊名稱 {#old-names}",
        "old_text": "這些名稱還能匯入，但每次呼叫都會發出警告，並指出新名稱。"
                    "之後的版本會把它們移除。",
        "old_head": "| 舊名稱 | 新名稱 |",
    },
}
#: The kind cell of a generated row, as written before it is translated.
_KIND = re.compile(r"(?<=\] \| )(function|class|constant)(?= \| )")


def _nav_names(config, language: str) -> Dict[str, str]:
    """English page titles to their names in *language*, from the i18n
    plugin's nav_translations in mkdocs.yml, so a page is named one way in
    the navigation and in the overview."""
    if language == "en":
        return {}
    plugin = config["plugins"].get("i18n")
    for lang in getattr(getattr(plugin, "config", None), "languages", []) or []:
        if lang.locale == language:
            return dict(lang.nav_translations or {})
    return {}


#: A Chinese character or full-width punctuation mark.
_CJK = re.compile("[　-〿一-鿿＀-￯]")
#: A line that opens a block of its own, so nothing is joined onto it.
_OPENS = re.compile(r"[ \t]*(?:[-*+]\s|\d+\.\s|[|#>:<]|!!!|\?\?\?|===|```|~~~)")
#: A line that ends a block, so the next line is never joined onto it.
_CLOSED = re.compile(r"[ \t]*(?:[|#<]|!!!|\?\?\?|===|```|~~~)")
#: A Han character, which the pages set apart from Latin text by a space.
_HAN = re.compile("[一-鿿]")
#: Latin text: a letter, a digit, or the backtick that opens inline code.
_LATIN = re.compile("[A-Za-z0-9`]")


def _edge(line: str, end: bool) -> str:
    """The character a reader sees at the end (or start) of a line, past a
    link's target, its brackets and emphasis marks."""
    if end:
        line = re.sub(r"\]\([^)]*\)$|\]\[[^\]]*\]$", "", line.rstrip())
        return line.rstrip("*_]")[-1:]
    return line.lstrip().lstrip("[*_")[:1]


def _join_cjk_lines(markdown: str) -> str:
    """Join the lines of a Chinese paragraph. A browser renders a line break
    as a space, and next to a Chinese character that space shows; the sources
    stay wrapped at 80 columns for the diffs. Where a break falls between a
    Chinese character and a Latin word, a number or inline code, the join
    keeps the space the pages put there. Fenced code is left alone."""
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
            a, b = _edge(prev, True), _edge(stripped, False)
            mixed = (_HAN.match(a) and _LATIN.match(b)) or (_LATIN.match(a) and _HAN.match(b))
            out[-1] = prev.rstrip() + (" " if mixed else "") + stripped
            continue
        out.append(line)
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files, **kwargs) -> str:
    chinese = page.file.src_uri.endswith(".zh-TW.md")
    if chinese:
        markdown = _join_cjk_lines(markdown)
    if MARKER not in markdown:
        return markdown
    return markdown.replace(MARKER, _overview(config, "zh-TW" if chinese else "en"))
