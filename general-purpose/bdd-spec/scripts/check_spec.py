#!/usr/bin/env python3
"""Lint a bdd-spec Markdown document for structural completeness.

Usage:
    python3 check_spec.py SPEC.md [--max-lines N]

Exit codes: 0 = every check passed (warnings allowed), 1 = at least one
failure, 2 = usage error or unreadable file.

FAIL checks
  - line count <= --max-lines (default 1000)
  - the 12 required top-level sections exist once each, in order 0..11,
    written as "## N. Title"
  - section 0 names Version, Date, Status, Author, and a change log
  - every AC-xx defined in section 4 carries exactly one valid priority tag
    (case and spacing inside the brackets are tolerated)
  - every AC-xx appears in sections 5, 7, 10 and 11; no AC is used but undefined
    (a section 4 row containing [DROPPED is exempt and reported as INFO)
  - a priority tag written next to an AC in section 5 matches section 4
  - the Stage 1 row(s) of section 10 (labelled "Stage 1", "1" or "fast lane";
    else the first row) hold every P0 case and nothing else; IDs are read
    from the Cases column only
  - AS/DEP/OQ tables in section 3 have Owner and Status columns and full rows
  - section 5 has Given / When / Then lines
  - section 6 has >= 1 mermaid block; every mermaid block in the file starts
    with a known diagram type; code fences are balanced
  - no images (![...] or <img) outside code fences

WARN checks
  - gaps in AC/OQ/AS/DEP numbering
  - an AC with fewer than two scenarios, or none whose title says error,
    edge or failure (naming convention, not a semantic check)
  - metadata status other than draft / approved
  - a resolved OQ whose Decision cell is empty
  - a section 11 entry that is neither a checkbox nor a table row
"""
import argparse
import re
import sys

SECTIONS = [
    (0, "metadata"),
    (1, "overview"),
    (2, "scope"),
    (3, "assumptions"),
    (4, "acceptance"),
    (5, "bdd|scenario"),
    (6, "diagram"),
    (7, "current behavi|gaps"),
    (8, "recommendation"),
    (9, "technical"),
    (10, "delivery|stages"),
    (11, "checklist|definition of done"),
]
HEADING_RE = re.compile(r"^##\s*(\d{1,2})[.):]?\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*([\w+-]*)")
ID_RES = {p: re.compile(r"\b%s-(\d{2,})\b" % p) for p in ("AC", "OQ", "AS", "DEP")}
PRIORITY_TAGS = {
    "[MUST / P0]": "P0",
    "[SHOULD / P1]": "P1",
    "[NICE TO HAVE / later]": "later",
}
TAG_RES = {  # tolerant of case and spacing: [must/p0] is accepted
    "P0": re.compile(r"\[\s*MUST\s*/\s*P0\s*\]", re.I),
    "P1": re.compile(r"\[\s*SHOULD\s*/\s*P1\s*\]", re.I),
    "later": re.compile(r"\[\s*NICE\s+TO\s+HAVE\s*/\s*LATER\s*\]", re.I),
}
ACCEPTED_TAGS = ", ".join(PRIORITY_TAGS)


def tags_in(line):
    return [p for p, r in TAG_RES.items() if r.search(line)]
MERMAID_TYPES = (
    "flowchart", "graph", "sequenceDiagram", "stateDiagram", "classDiagram",
    "erDiagram", "gantt", "journey", "gitGraph", "pie", "timeline", "mindmap",
    "quadrantChart", "xychart-beta", "block-beta", "C4Context", "C4Container",
    "C4Component", "C4Dynamic", "C4Deployment", "requirementDiagram",
    "sankey-beta", "packet-beta", "kanban", "architecture-beta", "zenuml",
)


class Report:
    def __init__(self):
        self.fails = []
        self.warns = []
        self.info = []

    def fail(self, msg):
        self.fails.append(msg)

    def warn(self, msg):
        self.warns.append(msg)


def parse_fences(lines):
    """Return (in_fence flags per line, list of (lang, start, end, body))."""
    in_fence = [False] * len(lines)
    blocks = []
    open_marker = None
    start = None
    lang = None
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if open_marker is None:
            if m:
                open_marker = m.group(1)
                lang = m.group(2).lower()
                start = i
                in_fence[i] = True
            continue
        in_fence[i] = True
        if m and m.group(1)[0] == open_marker[0] and len(m.group(1)) >= len(open_marker) and not m.group(2):
            blocks.append((lang, start, i, lines[start + 1:i]))
            open_marker = None
    unterminated = open_marker is not None
    return in_fence, blocks, unterminated, start


def find_sections(lines, in_fence, rep):
    found = {}
    order = []
    for i, line in enumerate(lines):
        if in_fence[i]:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        num = int(m.group(1))
        if num > 11:
            continue
        title = m.group(2)
        if num in found:
            rep.fail("line %d: section %d appears more than once (%r)" % (i + 1, num, title))
            continue
        found[num] = (i, title)
        order.append(num)
    for num, keyword in SECTIONS:
        if num not in found:
            rep.fail("section %d missing (heading should mention %r)" % (num, keyword.replace("|", " or ")))
        else:
            i, title = found[num]
            if not re.search(keyword, title, re.I):
                rep.fail("line %d: section %d heading %r does not mention %r" % (i + 1, num, title, keyword.replace("|", " or ")))
    if order != sorted(order):
        rep.fail("sections are out of order: %s" % order)
    # ranges: heading line .. line before next numbered heading
    ranges = {}
    starts = sorted((found[n][0], n) for n in found)
    for idx, (start, n) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        ranges[n] = (start, end)
    return ranges


def section_lines(lines, ranges, n):
    if n not in ranges:
        return []
    s, e = ranges[n]
    return list(enumerate(lines[s:e], start=s))


def ids_in(text, prefix):
    return set("%s-%s" % (prefix, m) for m in ID_RES[prefix].findall(text))


def check_metadata(lines, ranges, rep):
    text = "\n".join(l for _, l in section_lines(lines, ranges, 0)).lower()
    for field in ("version", "date", "status", "author"):
        if field not in text:
            rep.fail("section 0: missing %r" % field)
    if "change log" not in text and "changelog" not in text:
        rep.fail("section 0: missing change log")
    for _, l in section_lines(lines, ranges, 0):
        m = re.match(r"^\s*\|?\s*\**status\**\s*[|:]\s*(.+?)\s*\|?\s*$", l, re.I)
        if m and not re.search(r"\b(draft|approved)\b", m.group(1), re.I):
            rep.warn("section 0: status should be 'draft' or 'approved' (%r)" % l.strip())


def tables_in(sec):
    """Yield (header_cells, rows) for each pipe table in a list of (lineno, line)."""
    table = []
    for ln, line in list(sec) + [(-1, "")]:
        if line.strip().startswith("|"):
            table.append((ln, line))
        elif table:
            if len(table) >= 2:
                header = [c.strip() for c in table[0][1].strip().strip("|").split("|")]
                rows = [(ln, [c.strip() for c in l.strip().strip("|").split("|")]) for ln, l in table[2:]]
                yield table[0][0], header, rows
            table = []


def check_section3(lines, ranges, rep):
    sec = section_lines(lines, ranges, 3)
    seen = set()
    for header_ln, header, rows in tables_in(sec):
        hl = [h.lower() for h in header]
        id_rows = [(ln, cells) for ln, cells in rows if any(ID_RES[p].search(" ".join(cells)) for p in ("AS", "DEP", "OQ"))]
        if not id_rows:
            continue
        if not any("owner" in h for h in hl):
            rep.fail("line %d: section 3 table has no Owner column" % (header_ln + 1))
        if not any("status" in h for h in hl):
            rep.fail("line %d: section 3 table has no Status column" % (header_ln + 1))
        owner_i = next((i for i, h in enumerate(hl) if "owner" in h), None)
        status_i = next((i for i, h in enumerate(hl) if "status" in h), None)
        decision_i = next((i for i, h in enumerate(hl) if "decision" in h), None)
        for ln, cells in id_rows:
            rid = " ".join(cells)
            seen.update(ids_in(rid, "AS") | ids_in(rid, "DEP") | ids_in(rid, "OQ"))
            if len(cells) != len(header):
                rep.fail("line %d: row has %d cells, header has %d" % (ln + 1, len(cells), len(header)))
                continue
            for name, idx in (("owner", owner_i), ("status", status_i)):
                if idx is not None and not re.search(r"\w", cells[idx]):
                    rep.fail("line %d: empty %s cell" % (ln + 1, name))
            if status_i is not None and decision_i is not None:
                if re.search(r"resolved|decided|closed", cells[status_i], re.I) and not re.search(r"\w", cells[decision_i]):
                    rep.warn("line %d: status is resolved but Decision cell is empty" % (ln + 1))
    whole = "\n".join(lines)
    for p in ("AS", "DEP", "OQ"):
        used = ids_in(whole, p)
        undefined = sorted(used - seen)
        if undefined:
            rep.fail("%s referenced but not defined in a section 3 table: %s" % (p, ", ".join(undefined)))
    return seen


def check_acs(lines, ranges, rep):
    sec4 = section_lines(lines, ranges, 4)
    prio = {}
    dropped = set()
    for ln, line in sec4:
        if not line.strip().startswith("|"):
            continue  # only table rows define cases; prose may mention them
        m = ID_RES["AC"].search(line)
        if not m:
            continue
        ac = "AC-" + m.group(1)  # first ID defines the row; later IDs are cross-references
        tags = tags_in(line)
        if len(tags) != 1:
            rep.fail("line %d: %s must carry exactly one priority tag, found %d (accepted: %s)" % (ln + 1, ac, len(tags), ACCEPTED_TAGS))
            continue
        if ac in prio:
            rep.fail("line %d: %s defined twice in section 4" % (ln + 1, ac))
        prio[ac] = tags[0]
        if re.search(r"\[DROPPED\b", line):
            dropped.add(ac)
    if not prio:
        rep.fail("section 4: no acceptance cases (AC-xx rows) found")
        return prio
    if dropped:
        rep.info.append("dropped cases (kept in section 4, exempt elsewhere): %s" % ", ".join(sorted(dropped)))
    active = {a: p for a, p in prio.items() if a not in dropped}

    whole = "\n".join(lines)
    undefined = sorted(ids_in(whole, "AC") - set(prio))
    if undefined:
        rep.fail("AC referenced but not defined in section 4: %s" % ", ".join(undefined))

    # presence in 5, 7, 10, 11
    for n in (5, 7, 10, 11):
        text = "\n".join(l for _, l in section_lines(lines, ranges, n))
        present = ids_in(text, "AC")
        missing = sorted(set(active) - present)
        if missing:
            rep.fail("section %d: missing cases %s" % (n, ", ".join(missing)))

    # section 5: gherkin present, tag agreement, error coverage
    sec5 = section_lines(lines, ranges, 5)
    text5 = "\n".join(l for _, l in sec5)
    for kw in ("Given", "When", "Then"):
        if not re.search(r"^\s*%s\b" % kw, text5, re.M):
            rep.fail("section 5: no %r step found" % kw)
    for ln, line in sec5:
        for ac in ids_in(line, "AC"):
            for t in tags_in(line):
                if ac in prio and t != prio[ac]:
                    rep.fail("line %d: %s tagged %s in section 5 but %s in section 4" % (ln + 1, ac, t, prio[ac]))
    # scenario titles per AC: a "### AC-xx" heading opens a block; each
    # "Scenario:" / "Scenario Outline:" line belongs to the AC it names, else
    # to the open block
    titles = {}
    current = None
    for ln, line in sec5:
        m = ID_RES["AC"].search(line)
        if line.lstrip().startswith("#") and m:
            current = "AC-" + m.group(1)
        sm = re.match(r"\s*Scenario(?: Outline)?\s*:\s*(.*)", line)
        if sm:
            owner = ("AC-" + m.group(1)) if m else current
            if owner:
                titles.setdefault(owner, []).append(sm.group(1))
    for ac in sorted(active):
        t = titles.get(ac, [])
        if len(t) < 2:
            rep.warn("%s: %d scenario(s) in section 5; the contract asks for a happy path plus an error or edge scenario" % (ac, len(t)))
        elif not any(re.search(r"\b(error|edge|failure|fails?)\b", x, re.I) for x in t):
            rep.warn("%s: no scenario title says 'error', 'edge' or 'failure' (convention: 'Scenario: AC-xx error — ...')" % ac)

    # section 10: stage 1 == all P0
    sec10 = section_lines(lines, ranges, 10)
    stage1 = set()
    stage1_found = False
    first_row = None  # (cells, cases_i) fallback when no row is labelled Stage 1
    for header_ln, header, rows in tables_in(sec10):
        hl = [h.lower() for h in header]
        stage_i = next((i for i, h in enumerate(hl) if "stage" in h), 0)
        cases_i = next((i for i, h in enumerate(hl) if re.search(r"\bcases?\b|\bids?\b", h)), None)
        for ln, cells in rows:
            if first_row is None:
                first_row = (cells, cases_i)
            label = cells[stage_i] if stage_i < len(cells) else ""
            if re.search(r"stage\s*1\b|fast[ -]?lane|^\s*\**1\b", label, re.I):
                stage1_found = True
                src = cells[cases_i] if cases_i is not None and cases_i < len(cells) else " ".join(cells)
                stage1 |= ids_in(src, "AC")
    if not stage1_found and first_row is not None:
        cells, cases_i = first_row
        stage1_found = True
        stage1 = ids_in(cells[cases_i] if cases_i is not None and cases_i < len(cells) else " ".join(cells), "AC")
        rep.info.append("section 10: no row labelled 'Stage 1'; treating the first row as Stage 1")
    p0 = set(a for a, p in active.items() if p == "P0")
    if not stage1_found:
        rep.fail("section 10: no stage table found")
    else:
        if p0 - stage1:
            rep.fail("section 10: P0 cases missing from Stage 1: %s (add them to the Stage 1 row, or retag them [SHOULD / P1])" % ", ".join(sorted(p0 - stage1)))
        if stage1 - p0:
            rep.fail("section 10: non-P0 cases in Stage 1: %s (retag them [MUST / P0], or move them to a later stage)" % ", ".join(sorted(stage1 - p0)))
    if not p0:
        rep.fail("section 4: no [MUST / P0] cases — there is no fast lane")

    # section 11: checkbox or table row per AC
    for ln, line in section_lines(lines, ranges, 11):
        if ids_in(line, "AC") and not (re.search(r"\[( |x|X)\]", line) or line.strip().startswith("|")):
            rep.warn("line %d: section 11 entry is neither a checkbox nor a table row" % (ln + 1))
    return prio


def check_gaps(lines, rep):
    whole = "\n".join(lines)
    for p in ("AC", "OQ", "AS", "DEP"):
        nums = sorted(int(n) for n in ID_RES[p].findall(whole))
        if not nums:
            continue
        expected = list(range(1, max(nums) + 1))
        missing = sorted(set(expected) - set(nums))
        if missing:
            rep.warn("%s numbering has gaps: missing %s" % (p, ", ".join("%s-%02d" % (p, n) for n in missing)))


def check_mermaid(blocks, ranges, unterminated, fence_start, rep):
    if unterminated:
        rep.fail("line %d: code fence never closed" % (fence_start + 1))
    mermaid = [b for b in blocks if b[0] == "mermaid"]
    s6 = ranges.get(6)
    if s6 and not any(s6[0] <= b[1] < s6[1] for b in mermaid):
        rep.fail("section 6: no ```mermaid block")
    for lang, start, end, body in mermaid:
        first = None
        i = 0
        # skip mermaid YAML front matter
        if body and body[0].strip() == "---":
            i = 1
            while i < len(body) and body[i].strip() != "---":
                i += 1
            i += 1
        for line in body[i:]:
            s = line.strip()
            if s and not s.startswith("%%"):
                first = s
                break
        if first is None:
            rep.fail("line %d: empty mermaid block" % (start + 1))
            continue
        token = first.split()[0]
        if not any(token == t or token.startswith(t) for t in MERMAID_TYPES):
            rep.fail("line %d: mermaid block starts with %r, not a known diagram type" % (start + 1, token))


def check_images(lines, in_fence, rep):
    for i, line in enumerate(lines):
        if in_fence[i]:
            continue
        if "![" in line or "<img" in line.lower():
            rep.fail("line %d: image found; diagrams must be Mermaid source" % (i + 1))


def summarize(lines, prio, rep):
    whole = "\n".join(lines)
    counts = {"P0": 0, "P1": 0, "later": 0}
    for p in prio.values():
        counts[p] += 1
    rep.info.append("lines: %d" % len(lines))
    rep.info.append("acceptance cases: %d (P0 %d, P1 %d, later %d)" % (len(prio), counts["P0"], counts["P1"], counts["later"]))
    oq = len(ids_in(whole, "OQ"))
    resolved = len(re.findall(r"\|\s*OQ-\d+\s*\|[^\n]*\b(resolved|decided|closed)\b", whole, re.I))
    rep.info.append("open questions: %d (%d resolved)" % (oq, resolved))
    rep.info.append("mermaid blocks: %d" % whole.count("```mermaid"))


def main(argv):
    ap = argparse.ArgumentParser(description="Lint a bdd-spec Markdown document.")
    ap.add_argument("spec")
    ap.add_argument("--max-lines", type=int, default=1000)
    args = ap.parse_args(argv)
    try:
        with open(args.spec, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    rep = Report()
    if len(lines) > args.max_lines:
        rep.fail("file is %d lines; limit is %d" % (len(lines), args.max_lines))
    in_fence, blocks, unterminated, fence_start = parse_fences(lines)
    ranges = find_sections(lines, in_fence, rep)
    check_metadata(lines, ranges, rep)
    check_section3(lines, ranges, rep)
    prio = check_acs(lines, ranges, rep)
    check_gaps(lines, rep)
    check_mermaid(blocks, ranges, unterminated, fence_start, rep)
    check_images(lines, in_fence, rep)
    summarize(lines, prio, rep)

    for m in rep.fails:
        print("FAIL: " + m)
    for m in rep.warns:
        print("WARN: " + m)
    for m in rep.info:
        print("INFO: " + m)
    verdict = "PASS" if not rep.fails else "FAIL"
    print("%s — %d failure(s), %d warning(s): %s" % (verdict, len(rep.fails), len(rep.warns), args.spec))
    return 0 if not rep.fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
