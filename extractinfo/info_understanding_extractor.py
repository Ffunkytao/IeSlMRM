import json
import re
from typing import Any, Dict, List, Tuple, Callable, Optional

JsonDict = Dict[str, Any]


# ----------------------------- Utilities -----------------------------

def _safe_json_loads(s: str) -> Optional[JsonDict]:

    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to locate the outermost JSON object
    l = s.find('{')
    r = s.rfind('}')
    if l != -1 and r != -1 and r > l:
        chunk = s[l:r+1]
        try:
            return json.loads(chunk)
        except Exception:
            # Try to remove trailing commas / code fences / control chars
            chunk = re.sub(r'```.*?```', '', chunk, flags=re.S)
            chunk = re.sub(r',\s*([}\]])', r'\1', chunk)
            try:
                return json.loads(chunk)
            except Exception:
                return None
    return None


def _normalize_text(x: str) -> str:
    return re.sub(r'\s+', ' ', x.strip().lower())


def _token_set(x: str) -> set:
    return set(re.findall(r'[a-zA-Z0-9_\.%\-]+', _normalize_text(x)))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


# ----------------------------- Rule Base -----------------------------

def default_rule_base() -> List[JsonDict]:

    return [
        {
            "name": "Power-Energy-Time",
            "type": "physics",
            "enti": ["power", "energy", "time"],
            "unit": ["kW", "kWh", "h"],
            "formula": ["energy = power * time", "kWh = kW * h"]
        },
        {
            "name": "Speed-Distance-Time",
            "type": "physics",
            "enti": ["speed", "distance", "time"],
            "unit": ["km/h", "m/s", "km", "m", "h", "s"],
            "formula": ["distance = speed * time", "speed = distance / time"]
        },
        {
            "name": "Temperature-Delta",
            "type": "physics",
            "enti": ["temperature"],
            "unit": ["C", "°C", "K"],
            "formula": ["ΔT = T1 - T2"]
        },
        {
            "name": "Ratio-Percent",
            "type": "math",
            "enti": ["ratio", "percentage"],
            "unit": ["%", "ratio"],
            "formula": ["percentage = ratio * 100%"]
        }
    ]


# ------------------------ Similarity & Matching ----------------------

def semantic_match_score(candidate_rel: str, rule: JsonDict) -> float:

    cand_tokens = _token_set(candidate_rel)
    rule_text = " ".join([
        rule.get("name", ""),
        " ".join(rule.get("enti", [])),
        " ".join(rule.get("unit", [])),
        " ".join(rule.get("formula", []))
    ])
    rule_tokens = _token_set(rule_text)
    return _jaccard(cand_tokens, rule_tokens)


# --------------------------- Plan & Executor -------------------------

def plan_for_relation(candidate_rel: str, rule_base: List[JsonDict]) -> JsonDict:

    return {
        "relation": candidate_rel,
        "checks": [
            {"type": "unit_consistency", "detail": "units must be compatible with rule units"},
            {"type": "equation_validity", "detail": "equation/transform consistent with known formula"},
            {"type": "context_alignment", "detail": "relation tokens should appear in question/schema"}
        ],
        "rules_applied": [r.get("name") for r in rule_base]
    }


def _units_in_text(text: str) -> List[str]:
    # very light unit mining; extend as needed
    units = re.findall(r'\b(kwh|kw|km/h|m/s|km|m|h|s|%|c|°c|k)\b', text.lower())
    return list(set(units))


def _equation_like(text: str) -> bool:
    return bool(re.search(r'[\w\)\]]\s*=\s*[\w\(]', text))


def execute_plan(candidate_rel: str,
                 plan: JsonDict,
                 question: str,
                 db_info_text: str,
                 rule_base: List[JsonDict]) -> float:

    passed = 0
    total = len(plan["checks"]) or 1

    # 1) Unit consistency
    units_rel = set(_units_in_text(candidate_rel))
    units_ctx = set(_units_in_text(question + " " + db_info_text))
    rule_units = set()
    for r in rule_base:
        for u in r.get("unit", []):
            rule_units.add(u.lower())

    # condition: either relation has no units (neutral) or shares units with rule/ctx
    unit_ok = (not units_rel) or bool(units_rel & (rule_units | units_ctx))
    if unit_ok:
        passed += 1

    # 2) Equation validity
    eq_ok = _equation_like(candidate_rel)
    if not eq_ok:
        # try to see if relation tokens + rule formula share algebraic patterns
        # fallback heuristic: overlap with rule formulas tokens
        rel_tokens = _token_set(candidate_rel)
        score = max(_jaccard(rel_tokens, _token_set(" ".join(r.get("formula", [])))) for r in rule_base) if rule_base else 0.0
        eq_ok = score >= 0.2
    if eq_ok:
        passed += 1

    # 3) Context alignment: relation tokens should appear in question or db_info
    rel_tokens = _token_set(candidate_rel)
    ctx_tokens = _token_set(question + " " + db_info_text)
    ctx_ok = _jaccard(rel_tokens, ctx_tokens) >= 0.15
    if ctx_ok:
        passed += 1

    return passed / total


# ------------------------- LLM Draft Extraction ----------------------

DEFAULT_DRAFT_SYSTEM = (
    "You are a Text-to-SQL intelligence. Extract the following fields ONLY as strict JSON:\n"
    "{entities:[], relations:[], intent:\"\", reasoning_type:\"\", numerical_values:[], "
    "units:{}, required_tables:[], required_fields:[]}"
)

def llm_draft_extract(question: str,
                      db_info_text: str,
                      llm_fn: Optional[Callable[[str, str], str]] = None) -> JsonDict:
    
    if llm_fn is None:
        return {}
    user_msg = f"The question is:\n{question}\n\nDatabase info:\n{db_info_text}\n"
    raw = llm_fn(DEFAULT_DRAFT_SYSTEM, user_msg)
    js = _safe_json_loads(raw) or {}
    return js


# --------------------------- Main Pipeline ---------------------------

def build_R_init(draft_js: JsonDict) -> List[str]:

    rels = []
    for r in draft_js.get("relations", []) or []:
        if isinstance(r, str):
            rels.append(r)
        elif isinstance(r, dict):
            rels.append(json.dumps(r, ensure_ascii=False))
    # add signals from required_fields as weak relations
    for f in draft_js.get("required_fields", []) or []:
        rels.append(f"field_usage = {f}")
    # dedup while keeping order
    seen, out = set(), []
    for x in rels:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def filter_R_cand(R_init: List[str],
                  rule_base: List[JsonDict],
                  delta_match: float) -> List[str]:

    R_cand = []
    for r in R_init:
        best = max((semantic_match_score(r, rule) for rule in rule_base), default=0.0)
        if best > delta_match:
            R_cand.append(r)
    return R_cand


def extract_info_pipeline(
    question: str,
    db_info_text: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
    delta_match: float = 0.6,
    tau: float = 0.7,
    rule_base: Optional[List[JsonDict]] = None
) -> JsonDict:

    # 1) LLM draft extraction
    draft = llm_draft_extract(question, db_info_text, llm_fn=llm_fn)

    # ensure tuple components exist
    info_tuple = {
        "i": draft.get("intent", ""),
        "r": draft.get("relations", []) or [],
        "E": draft.get("entities", []) or [],
        "R": draft.get("required_fields", []) or [],
        "N": draft.get("numerical_values", []) or [],
        "U": draft.get("units", {}) or {},
        "P": draft.get("required_tables", []) or []  # treat as pseudo patterns/templates
    }

    # 2) 𝓡_init
    R_init = build_R_init(draft)

    # 3) 𝓡_cand via rule matching
    rules = rule_base if rule_base is not None else default_rule_base()
    R_cand = filter_R_cand(R_init, rules, delta_match=delta_match)

    # 4) Plan & Execute
    plans: List[JsonDict] = []
    scores: List[Tuple[str, float]] = []
    for rel in R_cand:
        P_i = plan_for_relation(rel, rules)
        s_i = execute_plan(rel, P_i, question, db_info_text, rules)
        plans.append(P_i)
        scores.append((rel, s_i))

    # 5) Filtering by τ
    R_high = [rel for rel, s in scores if s > tau]

    return {
        "R_tuple": info_tuple,           # (i, r, E, R, N, U, P)
        "R_init": R_init,                # initial relations from draft
        "R_cand": R_cand,                # after δ_match
        "plans": plans,                  # generated plans
        "scores": scores,                # (relation, s_i∈[0,1])
        "R_high": R_high,                # filtered high-quality relations
        "params": {"delta_match": delta_match, "tau": tau}
    }
