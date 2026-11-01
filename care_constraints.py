"""
Symbol-table-based constraint decoding -- iteration 1 (static constraints only).

This is the starting point for the GPU project. It contains three things:

  1. Symbol table loading (`build_symbol_table`, `SymbolTable`) -- the Python
     side of GroundedSymbolTable.java. One table per bug, keyed on the hole's
     line number, cached as JSON.

  2. Site extraction and checking (`extract_sites`, `StaticConstraints.check`)
     -- given a partially generated patch, find the positions that the symbol
     table can speak about and decide whether the names used there are legal.

  3. Token-level enforcement with LAZY WIDENING (`make_logits_processor`) --
     the baseline the project asks you to beat: check the Top-64 tokens, widen
     to Top-1024 only if too few survive, fail open if even that finds nothing.

Design rule: fail open. A position is constrained only when the symbol table
resolves it (known receiver, enum-typed enclosing switch, no wildcard imports
for `new`). Anything unresolved decodes freely, so the constraint never removes
a patch that javac would have accepted.

Iteration >= 2 (failure-adaptive constraints learned from compile/test output)
is not part of this release.
"""

import json
import re
import subprocess
from pathlib import Path

IDENT = r"[A-Za-z_$][\w$]*"
_IDENT_CHAR = re.compile(r"[\w$]")

# java.lang types are always visible without an import
JAVA_LANG = {
    "Object", "String", "StringBuilder", "StringBuffer", "Integer", "Long", "Short",
    "Byte", "Double", "Float", "Boolean", "Character", "Number", "Math", "System",
    "Thread", "Exception", "RuntimeException", "Error", "Throwable",
    "IllegalArgumentException", "IllegalStateException", "NullPointerException",
    "IndexOutOfBoundsException", "ArrayIndexOutOfBoundsException",
    "UnsupportedOperationException", "ClassCastException", "ArithmeticException",
    "NumberFormatException", "CloneNotSupportedException", "AssertionError",
    "StringIndexOutOfBoundsException", "NegativeArraySizeException",
}
PRIMITIVES = {"int", "long", "short", "byte", "double", "float", "boolean", "char"}


# --------------------------------------------------------------------------
# 1. Symbol table
# --------------------------------------------------------------------------

def hole_line_number(src_text, span, prefix_lines):
    """1-based line of the hole in the checked-out file.

    `span` is the (start, end) character offset of the buggy function inside
    `src_text`; `prefix_lines` are the function's lines before the hole.
    """
    return src_text[:span[0]].count("\n") + 1 + len(prefix_lines)


def load_symbol_table(tag, cache_dir):
    """Read a pre-computed symbol table for bug `tag` (e.g. "Chart_1").

    The release ships all 166 tables under symbol_cache/, so the common case
    needs neither Java nor a Defects4J checkout.
    """
    cache = Path(cache_dir) / f"{tag}.v2.json"
    if not cache.exists():
        return {"error": f"no cached symbol table at {cache}"}
    return json.loads(cache.read_text())


def build_symbol_table(tag, d4j, work, src_path, hole_line, symtab_cp, cache_dir,
                       timeout=300):
    """Run GroundedSymbolTable.java for one hole, cached on disk.

    Only needed to regenerate a table or to add a bug. `d4j` is anything with
    an `export(field, work)` method returning `defects4j export -p <field>`;
    `symtab_cp` is the classpath holding com.example.GroundedSymbolTable
    (e.g. "java/target/symbol-table.jar").
    """
    cache = Path(cache_dir) / f"{tag}.v2.json"   # bump when the Java JSON changes
    if cache.exists():
        return json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cp = d4j.export("cp.compile", work)
    src_root = Path(work) / d4j.export("dir.src.classes", work)
    cmd = ["java", "-cp", symtab_cp, "com.example.GroundedSymbolTable",
           "--file", str(src_path), "--line", str(hole_line),
           "--source-root", str(src_root), "--classpath", cp,
           "--out", str(cache)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not cache.exists():
        return {"error": f"symbol table failed: {(r.stderr or '')[-300:]}"}
    return json.loads(cache.read_text())


class SymbolTable:
    """The parts of the JSON the decoder needs, in lookup-friendly form."""

    def __init__(self, table):
        self.ok = "error" not in table
        self.receivers = {}                       # normalized receiver text -> set(names)
        for key, r in (table.get("after_dot") or {}).items():
            names = set()
            for field in ("methods", "fields", "enum_constants"):
                for o in r.get(field) or []:
                    names.add(o["name"] if isinstance(o, dict) else str(o))
            if names:
                self.receivers[_norm(key)] = names
        scope = table.get("scope") or {}
        # Type names as receivers: Branch.ON_EX, Token.INC, Math.max.
        # Added after variables, so a variable with the same name wins.
        for tname, r in (scope.get("type_members") or {}).items():
            key = _norm(tname)
            if key in self.receivers:
                continue
            tnames = set()
            for field in ("static_methods", "static_fields", "enum_constants", "nested_types"):
                for o in r.get(field) or []:
                    tnames.add(o["name"] if isinstance(o, dict) else str(o))
            if tnames:
                # always legal after a type name: Foo.class, Outer.this, Iface.super.m()
                self.receivers[key] = tnames | {"class", "this", "super"}
        # `case` is constrained only inside an enum-typed switch
        self.case_constants = set(table.get("hole_switch_constants") or [])
        self.types = set(scope.get("types") or []) | JAVA_LANG | PRIMITIVES
        # wildcard imports hide type names, so `new` must stay unconstrained
        self.new_enabled = self.ok and not scope.get("wildcard_imports") \
            and bool(scope.get("types"))
        self.health = table.get("health", {})

    def summary(self):
        return {
            "ok": self.ok,
            "receivers": len(self.receivers),
            "member_names": sum(len(v) for v in self.receivers.values()),
            "types": len(self.types),
            "case_constants": len(self.case_constants),
            "new_enabled": self.new_enabled,
            "resolution_rate": self.health.get("resolution_rate"),
        }


def _norm(expr):
    return re.sub(r"\s+", "", expr)


# --------------------------------------------------------------------------
# 2. Site extraction and checking on (partial) generated text
# --------------------------------------------------------------------------

def _sanitize(text):
    """Blank out string/char literals and comments. Returns (clean, open_tail):
    open_tail is True when the text ends inside a literal or comment, in which
    case nothing at the end is constrained."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                return "".join(out), True
            out.append(c + " " * (j - i - 1) + c)
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            if j == -1:
                return "".join(out), True
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j == -1:
                return "".join(out), True
            out.append(" " * (j + 2 - i))
            i = j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out), False


def _receiver_before(s, dot):
    """Text of the receiver chain ending just before s[dot] == '.', e.g.
    'foo', 'this.plot', 'getPlot()', 'dataset.getRowKey(row)'. '' if none."""
    i = dot - 1
    while i >= 0 and s[i].isspace():
        i -= 1
    end = i + 1
    while i >= 0:
        if s[i] == ")":
            depth = 0
            while i >= 0:
                if s[i] == ")":
                    depth += 1
                elif s[i] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                i -= 1
            i -= 1
        j = i
        while j >= 0 and _IDENT_CHAR.match(s[j]):
            j -= 1
        if j == i:                      # no identifier here
            break
        i = j
        k = i
        while k >= 0 and s[k].isspace():
            k -= 1
        if k >= 0 and s[k] == ".":
            i = k - 1
            while i >= 0 and s[i].isspace():
                i -= 1
            continue
        break
    recv = s[i + 1:end].strip()
    if not recv or recv[0].isdigit():   # 1.5, .5
        return ""
    return recv


def extract_sites(text, final):
    """Constrainable sites in `text`: list of (kind, receiver, name, complete).
    kind in {'dot', 'case', 'new'}. `complete` is False only for a name that
    runs to the end of a non-final text (still being generated)."""
    s, open_tail = _sanitize(text)
    sites = []

    def complete_at(end):
        return end < len(s) or final or open_tail

    for m in re.finditer(r"\.\s*(" + IDENT + r")", s):
        recv = _receiver_before(s, m.start())
        if recv:
            sites.append(("dot", _norm(recv), m.group(1), complete_at(m.end(1))))
    for m in re.finditer(r"\bcase\s+(" + IDENT + r")", s):
        sites.append(("case", None, m.group(1), complete_at(m.end(1))))
    for m in re.finditer(r"\bnew\s+(" + IDENT + r")(\s*\.)?", s):
        if m.group(2):                  # qualified: new java.util.X / Outer.Inner
            continue
        sites.append(("new", None, m.group(1), complete_at(m.end(1))))
    return sites


class StaticConstraints:
    """Symbol-table constraints for one bug (one hole). Iteration 1: static only."""

    def __init__(self, table, use_static=True):
        self.sym = SymbolTable(table)
        self.use_static = use_static and self.sym.ok
        self.n_checks = 0                # how many check() calls were made
        self.n_rejects = 0

    # ---- static domain of a site (None = unconstrained) --------------------
    def _domain(self, kind, recv):
        if not self.use_static:
            return None
        if kind == "dot":
            return self.sym.receivers.get(recv)
        if kind == "case":
            return self.sym.case_constants or None
        if kind == "new":
            return self.sym.types if self.sym.new_enabled else None
        return None

    def check(self, text, final=False):
        """True if the generated prefix `text` is still legal.

        A COMPLETE name must be a member of its site's domain. An INCOMPLETE
        name (one that runs to the end of a partial generation) only has to be
        a prefix of some member, so that a legal name can still be reached by
        generating more tokens.
        """
        self.n_checks += 1
        for kind, recv, name, complete in extract_sites(text, final):
            dom = self._domain(kind, recv)
            if dom is None:
                continue                 # unresolved site: fail open
            if complete:
                if name not in dom:
                    self.n_rejects += 1
                    return False
            elif not any(c.startswith(name) for c in dom):
                self.n_rejects += 1
                return False
        return True


# --------------------------------------------------------------------------
# 3. Token-level enforcement
# --------------------------------------------------------------------------

def build_piece_table(tokenizer):
    """Surface text of every token id ('▁' -> ' ', <0xNN> -> that byte).
    None for special tokens, which are never constrained."""
    special = set(tokenizer.all_special_ids)
    pieces = []
    for i, tok in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        if i in special or tok is None:
            pieces.append(None)
        elif re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", tok):
            b = int(tok[3:5], 16)
            pieces.append(chr(b) if b < 128 else "�")
        else:
            pieces.append(tok.replace("▁", " "))
    return pieces


class RowFilter:
    """Pure-Python core: which of `cand_ids` may follow `gen_ids`.

    This is the function the project asks you to move to the GPU. It is called
    once per beam per decoding step, and it is where essentially all of the
    constraint-checking time goes: `len(cand_ids)` independent calls to
    `StaticConstraints.check`, each of which re-scans the generated text.
    """

    def __init__(self, pieces, constraints, stop_ids):
        self.pieces = pieces
        self.c = constraints
        self.stop_ids = set(stop_ids)
        self.fallbacks = 0               # steps where nothing survived
        self.checked = 0                 # candidate tokens examined
        self.seconds = 0.0               # time spent inside allowed()

    def text_of(self, gen_ids):
        return "".join(self.pieces[i] or "" for i in gen_ids)

    def allowed(self, gen_ids, cand_ids):
        import time
        t0 = time.perf_counter()
        text = self.text_of(gen_ids)
        out = []
        for i in cand_ids:
            if i in self.stop_ids:
                ok = self.c.check(text, final=True)
            elif self.pieces[i] is None:
                ok = True                # other special tokens: fail open
            else:
                ok = self.c.check(text + self.pieces[i])
            if ok:
                out.append(i)
        self.checked += len(cand_ids)
        self.seconds += time.perf_counter() - t0
        return out


def make_logits_processor(tokenizer, constraints, stop_ids, top_k=64, wide_k=1024,
                          min_allowed=20, pieces=None):
    """HF LogitsProcessor implementing LAZY WIDENING (the project's baseline).

    Per row (= per beam) and per step:

        1. take the Top-`top_k` tokens by score and check them;
        2. if fewer than `min_allowed` survive, retake the Top-`wide_k` and
           check those instead;
        3. if still nothing survives, fail open and return the row unchanged.

    Set `min_allowed >= 2 * num_beams`, or beam search runs out of beams.

    Known limitation (section 1 of the handout): a valid token ranked below
    `wide_k` can never be recovered, no matter how many times we widen. The
    fix is to check the whole vocabulary -- which is what tasks 2-4 are about.
    Replace the body of the per-row loop below with your full-vocabulary CPU
    and CUDA versions and keep this one as the baseline to compare against.
    """
    import torch
    from transformers import LogitsProcessor

    core = RowFilter(pieces or build_piece_table(tokenizer), constraints, stop_ids)

    class LazyWideningLogitsProcessor(LogitsProcessor):
        def __init__(self):
            self.prompt_len = None
            self.core = core
            self.steps = 0
            self.widenings = 0

        def set_prompt_len(self, n):
            """Where the completion starts, so `gen_ids` excludes the prompt."""
            self.prompt_len = n

        def __call__(self, input_ids, scores):
            if self.prompt_len is None:
                raise RuntimeError("call set_prompt_len() before generate()")
            self.steps += 1
            out = torch.full_like(scores, float("-inf"))
            vocab = scores.shape[1]
            for row in range(scores.shape[0]):
                gen = input_ids[row, self.prompt_len:].tolist()
                allowed = []
                for stage, k in enumerate((top_k, wide_k)):
                    ids = torch.topk(scores[row], min(k, vocab)).indices.tolist()
                    allowed = core.allowed(gen, ids)
                    if len(allowed) >= min_allowed:
                        break
                    if stage == 0:
                        self.widenings += 1
                if not allowed:
                    core.fallbacks += 1
                    out[row] = scores[row]
                    continue
                out[row, allowed] = scores[row, allowed]
            return out

        def stats(self):
            return {
                "steps": self.steps,
                "widenings": self.widenings,
                "fallbacks": core.fallbacks,
                "tokens_checked": core.checked,
                "check_seconds": round(core.seconds, 3),
                "constraint_calls": constraints.n_checks,
            }

    return LazyWideningLogitsProcessor()
