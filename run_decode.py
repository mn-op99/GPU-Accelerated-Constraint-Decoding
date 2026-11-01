#!/usr/bin/env python3
"""
Single-line APR decoding, iteration 1 -- the baseline for the GPU project.

For each Defects4J bug it builds the fill-in-the-middle prompt the LoRA was
fine-tuned on, loads that bug's pre-computed symbol table, and decodes the
missing line either unconstrained or under symbol-table constraints enforced
with lazy widening (Top-64, widening to Top-1024).

No Defects4J checkout and no javac are needed: the symbol tables and the
class-skeleton context blocks ship pre-computed. Candidates are scored by
exact match against the developer fix, which is enough to compare decoding
strategies. Plausibility (section 5 of the handout) needs the real test suite.

Examples
--------
    # inspect one prompt and its symbol table, no model, no GPU
    python run_decode.py --bug Chart_1 --dry-run

    # unconstrained baseline
    python run_decode.py --limit 10 --constraints none

    # lazy-widening constrained decoding
    python run_decode.py --limit 10 --constraints static --num-beams 10

    # the constraint-checking cost on its own
    python run_decode.py --limit 10 --constraints static --report-stats
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from care_constraints import (StaticConstraints, build_piece_table,
                              load_symbol_table, make_logits_processor)

HERE = Path(__file__).resolve().parent

DEFAULT_DATASET = HERE / "data" / "defects4j_single_line_with_idx_augmented.jsonl"
DEFAULT_ADAPTER = HERE / "lora_full_function_skeleton_4096"
DEFAULT_BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
DEFAULT_SYMBOL_CACHE = HERE / "symbol_cache"
DEFAULT_STATIC_CACHE = HERE / "static_context_cache"
CONTEXT_WINDOW = 4096                    # what this LoRA was trained at

FIM_PRE, FIM_SUF, FIM_MID, FIM_EOT = "<PRE>", "<SUF>", "<MID>", "<EOT>"
FIM_RE = re.compile(r"^<PRE> (.*?) <SUF> (.*?) <MID> (.*?) <EOT>$", re.S)


# --------------------------------------------------------------------------
# 1. Dataset and hole position
# --------------------------------------------------------------------------

def load_dataset(path, bugs=None, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if bugs:
        wanted = set(bugs)
        rows = [r for r in rows if f"{r['project']}_{r['bug_id']}" in wanted]
    if limit:
        rows = rows[:limit]
    return rows


def marked_split(entry):
    """Split buggy_function on the // Buggy Line Start/End markers."""
    lines = entry["buggy_function_marked"].splitlines()
    start = end = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "// Buggy Line Start":
            start = i
        elif s == "// Buggy Line End":
            end = i
    if start is None or end is None or end < start:
        raise ValueError("buggy_function_marked has no usable marker pair")
    prefix = lines[:start]
    region = lines[start + 1:end]
    suffix = lines[end + 1:]
    if "\n".join(prefix + region + suffix) != entry["buggy_function"]:
        raise ValueError("marked split does not reconstruct buggy_function")
    return prefix, region + suffix, len(region)


def fim_split(entry):
    """Canonical hole position: (prefix_lines, suffix_lines, drop).

    `suffix_lines` still CONTAINS the buggy line -- that is the format the
    model was fine-tuned on. `drop` is how many leading suffix lines the
    generated patch replaces: 1 for replace/delete, 0 for insert.
    """
    raw = entry.get("buggy_function_fim") or ""
    m = FIM_RE.match(raw)
    if m:
        pre_txt, suf_txt, _mid = m.groups()
        lines = entry["buggy_function"].splitlines()
        k = len(pre_txt.split("\n"))
        prefix, suffix = lines[:k], lines[k:]
        if "\n".join(prefix) == pre_txt and "\n".join(suffix) == suf_txt:
            buggy_line = entry.get("buggy_line") or ""
            drop = 1 if (buggy_line and suffix and suffix[0] == buggy_line) else 0
            return prefix, suffix, drop
    return marked_split(entry)


# --------------------------------------------------------------------------
# 2. Prompt construction (FIM, matching the fine-tuning format)
# --------------------------------------------------------------------------

def load_static_context(entry, cache_dir):
    """The cached class-skeleton block for this bug, '' if absent."""
    f = Path(cache_dir) / f"{entry['project']}_{entry['bug_id']}_skeleton.json"
    if not f.exists():
        return ""
    try:
        data = json.loads(f.read_text())
    except json.JSONDecodeError:
        return ""
    return data.get("skeleton", "") if isinstance(data, dict) else str(data)


def as_java_comment(text):
    return "\n".join("// " + l if l else "//" for l in text.splitlines())


def build_prompt(prefix_lines, suffix_lines, ctx_comment):
    pre_parts = [p for p in (ctx_comment, "\n".join(prefix_lines)) if p]
    prefix = "\n".join(pre_parts)
    suffix = "\n".join(suffix_lines)
    return f"{FIM_PRE} {prefix} {FIM_SUF} {suffix} {FIM_MID}"


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_prompt(tokenizer, prefix_lines, suffix_lines, ctx_comment, budget):
    """Build the prompt, dropping context lines from the tail until it fits.

    The buggy method itself is never trimmed: a syntactically truncated
    function is worse input than a long one. If the method alone overruns the
    window the prompt is sent over budget and that is reported.
    """
    ctx = ctx_comment
    prompt = build_prompt(prefix_lines, suffix_lines, ctx)
    if tokenizer is None or not budget:
        return prompt, {"prompt_tokens": None, "over_budget": False}
    for _ in range(16):
        n = count_tokens(tokenizer, prompt)
        if n <= budget:
            return prompt, {"prompt_tokens": n, "over_budget": False}
        if not ctx:
            return prompt, {"prompt_tokens": n, "over_budget": True}
        lines = ctx.splitlines()
        per_line = n / max(len(lines), 1)
        cut = max(1, int((n - budget) / max(per_line, 0.5)) + 1)
        lines = lines[:-cut] if cut < len(lines) else []
        ctx = "\n".join(lines + ["// ... (truncated)"]) if lines else ""
        prompt = build_prompt(prefix_lines, suffix_lines, ctx)
    return prompt, {"prompt_tokens": count_tokens(tokenizer, prompt),
                    "over_budget": True}


def clean_generation(text, max_lines=4):
    """Strip the training-time separators and cut down to the infilled line(s).

    Returns None for a runaway completion (not a single-line patch).
    """
    for tok in ("▁" + FIM_EOT, "▁" + FIM_PRE,
                "▁" + FIM_SUF, "▁" + FIM_MID,
                FIM_EOT, FIM_PRE, FIM_SUF, FIM_MID,
                "</s>", "<|endoftext|>", "<s>"):
        i = text.find(tok)
        if i != -1:
            text = text[:i]
    if text.startswith(" "):
        text = text[1:]              # the single separator space after <MID>
    text = text.rstrip()
    lines = text.split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    if max_lines and len(lines) > max_lines:
        return None
    return lines


def normalize_for_exact(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def first_nonempty_line(text):
    for l in (text or "").splitlines():
        if l.strip():
            return l
    return ""


# --------------------------------------------------------------------------
# 3. Model
# --------------------------------------------------------------------------

class Repairer:
    """Fine-tuned Code Llama + LoRA adapter, loaded for inference."""

    def __init__(self, adapter, base_model=None, dtype="auto", device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        adapter = str(adapter)
        is_adapter = (Path(adapter) / "adapter_config.json").exists()
        base = base_model or _recorded_base(adapter) or DEFAULT_BASE_MODEL

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if dtype == "auto":
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        if self.device == "cuda":
            # bf16 tensor cores for the matmuls, TF32 for whatever stays fp32.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(f"[model] base:    {base}")
        print(f"[model] adapter: {adapter if is_adapter else '(none)'}")

        tok_src = adapter if (Path(adapter) / "tokenizer_config.json").exists() else base
        self.tokenizer = load_tokenizer(tok_src, base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # transformers >= 4.56 renamed the from_pretrained dtype argument
        import inspect
        dtype_kw = ("dtype"
                    if "dtype" in inspect.signature(
                        AutoModelForCausalLM.from_pretrained).parameters
                    else "torch_dtype")
        kwargs = {dtype_kw: torch_dtype}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        if is_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            # folding the deltas into the base weights is much faster at inference
            model = model.merge_and_unload()
            print("[model] LoRA merged into base weights")
        if "device_map" not in kwargs:
            model = model.to(self.device)
        model.eval()
        self.model = model

        eot = self.tokenizer.encode(FIM_EOT, add_special_tokens=False)
        self.eot_id = eot[0] if len(eot) == 1 else None
        self.pieces = None               # built once, reused across bugs
        print(f"[model] device={self.device} dtype={torch_dtype} eot_id={self.eot_id}")

    def prompt_budget(self, max_new_tokens):
        return max(256, CONTEXT_WINDOW - max_new_tokens - 8)

    def stop_ids(self):
        return [i for i in (self.eot_id, self.tokenizer.eos_token_id) if i is not None]

    def piece_table(self):
        if self.pieces is None:
            self.pieces = build_piece_table(self.tokenizer)
        return self.pieces

    def generate(self, prompt, n, max_new_tokens, num_beams=1, temperature=0.0,
                 top_p=0.95, seed=0, logits_processor=None):
        torch = self.torch
        if num_beams > 1:
            # beam search is deterministic and returns the n best beams at once
            sampling = False
        elif temperature <= 0:
            # greedy would give n identical lines, and transformers rejects
            # num_return_sequences > 1 for it outright
            sampling, n = False, 1
        else:
            sampling = True

        torch.manual_seed(seed)
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=sampling,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if sampling:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        if num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
        if self.eot_id is not None:
            gen_kwargs["eos_token_id"] = [self.eot_id, self.tokenizer.eos_token_id]
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            # the processor needs to know where the completion starts
            logits_processor.set_prompt_len(enc["input_ids"].shape[1])
            gen_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
            gen_kwargs["renormalize_logits"] = True

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        in_len = enc["input_ids"].shape[1]
        return [self.tokenizer.decode(row[in_len:], skip_special_tokens=False)
                for row in out]


def load_tokenizer(src, base):
    """Load the checkpoint's tokenizer, with a fallback for old `tokenizers`.

    This checkpoint's tokenizer.json was written by tokenizers >= 0.20, whose
    `merges` are pairs (`[["a","b"], ...]`). tokenizers < 0.20 cannot parse
    that and dies with "data did not match any variant of untagged enum
    ModelWrapper". Rather than force an upgrade, rewrite a copy with the old
    `"a b"` merge spelling and load that; identical vocabulary either way.
    """
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(src)
    except Exception as ex:
        print(f"[tok] {src} failed to load ({type(ex).__name__}: "
              f"{str(ex)[:100]}); trying a legacy-merges copy")

    tj = Path(src) / "tokenizer.json"
    if tj.exists():
        import hashlib, shutil, tempfile
        # cached on a path derived from the checkpoint: rewritten once per machine
        key = hashlib.sha1(str(Path(src).resolve()).encode()).hexdigest()[:12]
        compat = Path(tempfile.gettempdir()) / f"tok_compat_{key}"
        if not (compat / "tokenizer.json").exists():
            data = json.loads(tj.read_text())
            merges = data.get("model", {}).get("merges")
            if merges and isinstance(merges[0], list):
                data["model"]["merges"] = [" ".join(m) for m in merges]
                compat.mkdir(parents=True, exist_ok=True)
                for f in ("tokenizer_config.json", "special_tokens_map.json",
                          "chat_template.jinja"):
                    if (Path(src) / f).exists():
                        shutil.copy(Path(src) / f, compat / f)
                (compat / "tokenizer.json").write_text(json.dumps(data))
        if (compat / "tokenizer.json").exists():
            print(f"[tok] loaded via legacy-merges copy at {compat} "
                  f"(upgrade `tokenizers` to >=0.20 to avoid this)")
            return AutoTokenizer.from_pretrained(str(compat))

    print(f"[tok] falling back to the base model tokenizer: {base}")
    return AutoTokenizer.from_pretrained(base)


def _recorded_base(adapter):
    """The base model named in adapter_config.json, if it is usable here.

    A local path baked in at training time will not exist on another machine,
    so it is ignored in favour of DEFAULT_BASE_MODEL. A Hub id (no leading
    path) is taken as given.
    """
    cfg = Path(adapter) / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        recorded = json.loads(cfg.read_text()).get("base_model_name_or_path")
    except json.JSONDecodeError:
        return None
    if not recorded:
        return None
    if Path(recorded).exists() or not recorded.startswith("/"):
        return recorded
    print(f"[model] adapter_config.json points at {recorded!r}, which does not "
          f"exist here; using {DEFAULT_BASE_MODEL!r}")
    return None


# --------------------------------------------------------------------------
# 4. One bug
# --------------------------------------------------------------------------

def decode_one(entry, repairer, args):
    tag = f"{entry['project']}_{entry['bug_id']}"
    prefix, suffix, drop = fim_split(entry)
    ctx = load_static_context(entry, args.static_cache)
    ctx_comment = as_java_comment(ctx) if (ctx and args.context) else ""

    record = {"bug": tag, "drop": drop, "candidates": [], "exact_match": False}

    constraints = proc = None
    if args.constraints != "none":
        table = load_symbol_table(tag, args.symbol_cache)
        constraints = StaticConstraints(table)
        record["symbol_table"] = constraints.sym.summary()
        if not constraints.sym.ok:
            print(f"    [sym] {tag}: {table.get('error', 'no table')}; "
                  f"decoding unconstrained")
            constraints = None

    tokenizer = repairer.tokenizer if repairer else None
    budget = repairer.prompt_budget(args.max_new_tokens) if repairer else None
    prompt, fit = fit_prompt(tokenizer, prefix, suffix, ctx_comment, budget)
    record["prompt_tokens"] = fit["prompt_tokens"]
    if fit["over_budget"]:
        print(f"    [fit] {tag}: method alone is {fit['prompt_tokens']} tokens, "
              f"over the {budget}-token budget; sending it whole")

    if args.dry_run:
        record["prompt"] = prompt
        return record

    if constraints is not None:
        proc = make_logits_processor(
            repairer.tokenizer, constraints, repairer.stop_ids(),
            top_k=args.top_k, wide_k=args.wide_k,
            min_allowed=max(20, 2 * args.num_beams),
            pieces=repairer.piece_table())

    t0 = time.perf_counter()
    raw = repairer.generate(prompt, n=args.num_candidates,
                            max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams,
                            temperature=args.temperature, top_p=args.top_p,
                            seed=args.seed, logits_processor=proc)
    record["decode_seconds"] = round(time.perf_counter() - t0, 3)
    if proc is not None:
        record["constraint_stats"] = proc.stats()

    gold = normalize_for_exact(first_nonempty_line(entry["fixed_line"]))
    for text in raw:
        lines = clean_generation(text, args.max_patch_lines)
        if lines is None:
            record["candidates"].append({"patch": None, "reason": "runaway"})
            continue
        patch = "\n".join(lines)
        exact = normalize_for_exact(first_nonempty_line(patch)) == gold
        record["candidates"].append({"patch": patch, "exact_match": exact})
        record["exact_match"] = record["exact_match"] or exact
    return record


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--bug", action="append", default=None,
                    help="restrict to Project_Id, repeatable (e.g. --bug Chart_1)")
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--base-model", default=None,
                    help=f"default: adapter_config.json, else {DEFAULT_BASE_MODEL}")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])

    ap.add_argument("--constraints", default="static", choices=["none", "static"],
                    help="none = unconstrained; static = symbol table + lazy widening")
    ap.add_argument("--top-k", type=int, default=64,
                    help="candidates checked per beam per step before widening")
    ap.add_argument("--wide-k", type=int, default=1024,
                    help="candidates checked after widening")
    ap.add_argument("--symbol-cache", default=str(DEFAULT_SYMBOL_CACHE))
    ap.add_argument("--static-cache", default=str(DEFAULT_STATIC_CACHE))
    ap.add_argument("--no-context", dest="context", action="store_false",
                    help="omit the cached class skeleton from the prompt")

    ap.add_argument("--num-candidates", type=int, default=10)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (one candidate)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-patch-lines", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt and symbol-table stats, load no model")
    ap.add_argument("--report-stats", action="store_true",
                    help="print constraint-checking counters per bug")
    ap.add_argument("--out", default=None, help="write the records to this JSON file")
    args = ap.parse_args()

    rows = load_dataset(args.dataset, args.bug, args.limit)
    if not rows:
        print("no bugs selected", file=sys.stderr)
        return 1
    print(f"[data] {len(rows)} bug(s) from {args.dataset}")

    repairer = None if args.dry_run else Repairer(
        args.adapter, args.base_model, args.dtype, args.device)

    records, n_exact, total_check = [], 0, 0.0
    for i, entry in enumerate(rows, 1):
        tag = f"{entry['project']}_{entry['bug_id']}"
        print(f"\n[{i}/{len(rows)}] {tag}")
        try:
            rec = decode_one(entry, repairer, args)
        except Exception as ex:
            print(f"    [error] {type(ex).__name__}: {ex}", file=sys.stderr)
            records.append({"bug": tag, "error": f"{type(ex).__name__}: {ex}"})
            continue
        records.append(rec)

        if rec.get("symbol_table"):
            s = rec["symbol_table"]
            print(f"    [sym] {s['receivers']} receivers, {s['member_names']} member "
                  f"names, {s['types']} types, resolution={s['resolution_rate']}")
        if args.dry_run:
            print(f"    [prompt] {rec['prompt_tokens']} tokens")
            print("-" * 70)
            print(rec["prompt"])
            print("-" * 70)
            continue

        n_exact += bool(rec["exact_match"])
        if rec.get("constraint_stats"):
            cs = rec["constraint_stats"]
            total_check += cs["check_seconds"]
            if args.report_stats:
                print(f"    [cons] {cs['steps']} steps, {cs['widenings']} widenings, "
                      f"{cs['fallbacks']} fallbacks, {cs['tokens_checked']} tokens "
                      f"checked in {cs['check_seconds']}s")
        print(f"    [gen] {rec['decode_seconds']}s, "
              f"exact_match={rec['exact_match']}")
        for c in rec["candidates"][:3]:
            mark = "*" if c.get("exact_match") else " "
            print(f"      {mark} {c.get('patch') or '<' + c.get('reason', '?') + '>'}")

    if not args.dry_run:
        print(f"\n=== {n_exact}/{len(rows)} bugs with an exact-match candidate ===")
        if total_check:
            print(f"=== {round(total_check, 2)}s total in constraint checking ===")
    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
