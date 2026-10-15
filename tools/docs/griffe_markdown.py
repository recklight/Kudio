# -*- coding: utf-8 -*-
"""Read kudio's docstrings the way ``help()`` reads them, and hand mkdocstrings
Markdown.

The docstrings are written for ``help()`` and an editor's hover: Sphinx roles
(:func:`file_load`), ``>>>`` examples, ``::`` literal blocks and ``#:``
comments above module constants. mkdocstrings treats a docstring as Markdown,
where a role prints as raw text, every ``>>>`` line opens another nested block
quote, and a ``#:`` comment does not exist at all. Rewriting the source for
the site would make ``help()`` worse, so the translation happens here, at
build time, and only the site sees it.

Enabled from mkdocs.yml as a griffe extension of the python handler.
"""
from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional

import griffe

#: ``:func:`name``` and friends, with Sphinx's ``~`` for "show the last part".
_ROLE = re.compile(r":(func|class|meth|attr|data|mod|exc|obj):`(~?)([^`]+)`")
#: An opening or closing Markdown fence, with whatever indentation it has.
_FENCE = re.compile(r"^\s*(```|~~~)")
#: Doctest directives are for the test runner, not the reader.
_DIRECTIVE = re.compile(r"\s*#\s*doctest:.*$")
#: A block that looks like code gets Python highlighting; anything else, none.
_CODE_HINT = re.compile(r"\b(import|def|return|raise|except|for|with)\b|=|\(")


def _walk(obj: griffe.Object) -> Iterator[griffe.Object]:
    yield obj
    for member in getattr(obj, "members", {}).values():
        if member.is_alias:
            continue
        yield from _walk(member)


def _final(obj) -> Optional[griffe.Object]:
    """Follow an alias to what it names; None when that is outside kudio."""
    if obj is None or not obj.is_alias:
        return obj
    try:
        return obj.final_target
    except (griffe.AliasResolutionError, griffe.CyclicAliasError):
        return None


def _lookup(root: griffe.Module, path: str) -> Optional[griffe.Object]:
    parts = path.split(".")
    if parts[0] != root.name:
        return None
    obj: Optional[griffe.Object] = root
    for part in parts[1:]:
        obj = _final(obj)
        if obj is None or part not in getattr(obj, "members", {}):
            return None
        obj = obj.members[part]
    return _final(obj)


def _resolve(target: str, here: griffe.Object, root: griffe.Module,
             by_name: Dict[str, List[griffe.Object]]) -> Optional[griffe.Object]:
    """Find what a role names, the way Sphinx looks: outward from the
    docstring, then by a name only one object in kudio has."""
    candidates = []
    if target.split(".")[0] == root.name:
        candidates.append(target)
    else:
        scope: Optional[griffe.Object] = here
        while scope is not None:
            candidates.append(f"{scope.path}.{target}")
            scope = scope.parent
    for path in candidates:
        found = _lookup(root, path)
        if found is not None:
            return found
    first, _, rest = target.partition(".")
    owners = by_name.get(first, [])
    if len(owners) == 1:
        return _lookup(root, f"{owners[0].path}.{rest}") if rest else owners[0]
    return None


def _rendered(obj: griffe.Object, root: griffe.Module, skip: tuple) -> bool:
    """Would the API reference have an anchor for *obj*? Only public objects
    under a documented module that carry a docstring get one."""
    parts = obj.path.split(".")
    if parts[0] != root.name or any(p.startswith("_") for p in parts[1:]):
        return False
    if obj.is_module and obj.is_init_module:
        return False                    # packages are not rendered, only modules
    if any(obj.path == m or obj.path.startswith(m + ".") for m in skip):
        return False
    return bool(obj.docstring and obj.docstring.value.strip())


class DocstringMarkdown(griffe.Extension):
    """Translate kudio's reStructuredText habits into Markdown for the site.

    :param skip: modules the API reference does not render, so nothing links
        into them (the command line has a page of its own).
    """

    def __init__(self, skip: Optional[List[str]] = None) -> None:
        super().__init__()
        self.skip = tuple(skip or ())

    # -- #: comments become attribute docstrings ------------------------------

    def on_attribute_instance(self, *, attr: griffe.Attribute, **kwargs) -> None:
        if attr.docstring is not None or attr.lineno is None or not attr.filepath:
            return
        if attr.filepath not in attr.lines_collection:
            return
        lines = attr.lines_collection[attr.filepath]
        comment: List[str] = []
        row = attr.lineno - 2                      # the line above, 0-based
        while row >= 0 and lines[row].strip().startswith("#:"):
            comment.insert(0, lines[row].strip()[2:].strip())
            row -= 1
        if comment:
            attr.docstring = griffe.Docstring(
                "\n".join(comment), lineno=row + 2, parent=attr)

    # -- everything else, once the whole package is loaded --------------------

    def on_package(self, *, pkg: griffe.Module, **kwargs) -> None:
        if pkg.name != "kudio":
            return
        by_name: Dict[str, List[griffe.Object]] = {}
        objects = list(_walk(pkg))
        for obj in objects:
            by_name.setdefault(obj.name, []).append(obj)
        for obj in objects:
            if obj.docstring is not None:
                obj.docstring.value = self._translate(obj.docstring.value, obj,
                                                      pkg, by_name)

    def _translate(self, text: str, here: griffe.Object, root: griffe.Module,
                   by_name: Dict[str, List[griffe.Object]]) -> str:
        lines = _literal_blocks(_doctests(text.split("\n")))

        def role(match: "re.Match[str]") -> str:
            short, target = match.group(2) == "~", match.group(3).strip()
            shown = target.rsplit(".", 1)[-1] if short else target
            found = _resolve(target, here, root, by_name)
            if found is not None and _rendered(found, root, self.skip):
                return f"[`{shown}`][{found.path}]"
            return f"`{shown}`"

        out, fenced = [], False
        for line in lines:
            if _FENCE.match(line):
                fenced = not fenced
            out.append(line if fenced else _ROLE.sub(role, line))
        return "\n".join(out)


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _doctests(lines: List[str]) -> List[str]:
    """``>>>`` examples become fenced ``pycon`` blocks. A doctest block runs to
    the first blank line, which is also where doctest itself stops reading."""
    out: List[str] = []
    i, fenced = 0, False
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            fenced = not fenced
        if fenced or not line.lstrip().startswith(">>>"):
            out.append(line)
            i += 1
            continue
        pad = _indent(line)
        block = []
        while i < len(lines) and lines[i].strip() and lines[i].startswith(pad):
            block.append(_DIRECTIVE.sub("", lines[i][len(pad):]).rstrip())
            i += 1
        if out and out[-1].strip():
            out.append("")
        out += [pad + "```pycon"] + [pad + b for b in block] + [pad + "```"]
        if i < len(lines) and lines[i].strip():
            out.append("")
    return out


def _literal_blocks(lines: List[str]) -> List[str]:
    """``text::`` followed by an indented block is reStructuredText for "a
    colon, then this verbatim"; Markdown gets the colon and a fence."""
    out: List[str] = []
    i, fenced = 0, False
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            fenced = not fenced
        if fenced or not line.rstrip().endswith("::"):
            out.append(line)
            i += 1
            continue
        head = line.rstrip()[:-2]
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j == len(lines) or len(_indent(lines[j])) <= len(_indent(line)):
            out.append(line)                       # not a literal block after all
            i += 1
            continue
        if head.strip():
            out.append(head.rstrip() + ("" if head.endswith(" ") else ":"))
        inner = _indent(lines[j])
        block = []
        k = j
        while k < len(lines) and (not lines[k].strip() or lines[k].startswith(inner)):
            block.append(lines[k][len(inner):])
            k += 1
        while block and not block[-1].strip():
            block.pop()
        lang = "python" if any(_CODE_HINT.search(b) for b in block) else "text"
        pad = _indent(line)
        out += ["", pad + "```" + lang] + [pad + b for b in block] + [pad + "```", ""]
        i = k
    return out
