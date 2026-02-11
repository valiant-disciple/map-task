"""
Survey Processing Pipeline
- NASA-TLX scoring (raw + weighted)
- PSMM scoring (task SMM, team SMM subscales)
- Trial success data
- Demographics summary
"""

import os
import json
import numpy as np


# ── NASA-TLX ────────────────────────────────────────────────────────

TLX_DIMENSIONS = ["mental", "physical", "temporal", "performance", "effort", "frustration"]

def score_tlx(tlx_data: dict) -> dict:
    """
    Score a single NASA-TLX response.
    Input: dict with keys 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration'
    Values are 0-100 (21-point scale mapped to 0-100).
    Returns raw scores + overall workload (unweighted mean).
    """
    scores = {}
    valid = []
    for dim in TLX_DIMENSIONS:
        val = tlx_data.get(dim)
        if val is not None:
            scores[dim] = float(val)
            valid.append(float(val))
        else:
            scores[dim] = None

    scores["overall_workload"] = round(np.mean(valid), 2) if valid else None
    return scores


def process_tlx(tlx_list: list) -> dict:
    """Process list of TLX submissions for a role. Usually just one per trial."""
    if not tlx_list:
        return {"error": "No TLX data"}

    # Take the last submission (in case of duplicates)
    raw = tlx_list[-1] if isinstance(tlx_list[-1], dict) else {}
    scored = score_tlx(raw)
    return scored


# ── PSMM (Perceived Shared Mental Models) ───────────────────────────

PSMM_FACTORS = {
    "task": [1, 2, 3, 4],    # Items 1-4: Task SMM
    "team": [5, 6, 7, 8],    # Items 5-8: Team SMM
}

def score_psmm(psmm_rows: list) -> dict:
    """
    Score PSMM responses.
    Input: list of {factor, itemNum, value} dicts (1-7 Likert scale).
    Returns subscale means and overall score.
    """
    if not psmm_rows:
        return {"error": "No PSMM data"}

    # Build item map
    items = {}
    for row in psmm_rows:
        item_num = row.get("itemNum")
        value = row.get("value")
        factor = row.get("factor", "unknown")
        if item_num is not None and value is not None:
            items[int(item_num)] = {"value": float(value), "factor": factor}

    result = {"items": items}

    # Subscale scores
    for factor, item_nums in PSMM_FACTORS.items():
        vals = [items[n]["value"] for n in item_nums if n in items]
        result[f"{factor}_smm_mean"] = round(np.mean(vals), 3) if vals else None
        result[f"{factor}_smm_items"] = len(vals)

    # Overall
    all_vals = [v["value"] for v in items.values()]
    result["overall_smm_mean"] = round(np.mean(all_vals), 3) if all_vals else None

    return result


# ── Trial Success ───────────────────────────────────────────────────

def process_trial_success(events: list) -> dict:
    """Extract trial success data from events."""
    success_events = [e for e in events if e.get("type") == "trial_success"]
    if not success_events:
        return {"reported": False}

    data = success_events[-1].get("payload", {})
    return {
        "reported": True,
        "target_reached": data.get("targetReached"),
        "path_confidence": data.get("pathConfidence"),
        "note": data.get("note", ""),
    }


# ── Demographics ────────────────────────────────────────────────────

def extract_demographics(events: list) -> dict:
    """Extract demographics data from event log."""
    demo_events = [e for e in events if e.get("type") == "demographics"]
    result = {}
    for de in demo_events:
        payload = de.get("payload", {})
        role = de.get("role") or payload.get("role", "unknown")
        result[role] = {
            "age": payload.get("age"),
            "gender": payload.get("gender"),
            "handedness": payload.get("handedness"),
            "nativeLanguage": payload.get("nativeLanguage"),
            "englishFluency": payload.get("englishFluency"),
            "partnerFamiliarity": payload.get("partnerFamiliarity"),
            "priorMapTask": payload.get("priorMapTask"),
        }
    return result


# ── Debrief ─────────────────────────────────────────────────────────

def extract_debrief(events: list) -> dict:
    """Extract debrief data from event log."""
    debrief_events = [e for e in events if e.get("type") == "debrief_submit"]
    if not debrief_events:
        return {}
    payload = debrief_events[-1].get("payload", {})
    return {
        "strategy": payload.get("strategy", ""),
        "communication": payload.get("communication", ""),
        "challenges": payload.get("challenges", ""),
        "suggestions": payload.get("suggestions", ""),
    }


# ── Full Trial Survey Processing ────────────────────────────────────

def process_trial_surveys(trial_dir: str) -> dict:
    """Process all survey data for a single trial."""
    result = {}

    # Events file for trial success
    events_path = os.path.join(trial_dir, "events.json")
    events = []
    if os.path.exists(events_path):
        with open(events_path) as f:
            events = json.load(f)
    result["trial_success"] = process_trial_success(events)

    # TLX
    for role in ["director", "matcher"]:
        tlx_path = os.path.join(trial_dir, f"tlx_{role}.json")
        if os.path.exists(tlx_path):
            with open(tlx_path) as f:
                tlx_data = json.load(f)
            result[f"tlx_{role}"] = process_tlx(tlx_data)
        else:
            result[f"tlx_{role}"] = {"error": "No TLX file"}

    # PSMM
    for role in ["director", "matcher"]:
        psmm_path = os.path.join(trial_dir, f"psmm_{role}.json")
        if os.path.exists(psmm_path):
            with open(psmm_path) as f:
                psmm_data = json.load(f)
            result[f"psmm_{role}"] = score_psmm(psmm_data)
        else:
            result[f"psmm_{role}"] = {"error": "No PSMM file"}

    # Save
    json_path = os.path.join(trial_dir, "surveys_processed.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  [surveys] TLX dir={result.get('tlx_director', {}).get('overall_workload', '?')} "
          f"mat={result.get('tlx_matcher', {}).get('overall_workload', '?')} | "
          f"PSMM dir={result.get('psmm_director', {}).get('overall_smm_mean', '?')} "
          f"mat={result.get('psmm_matcher', {}).get('overall_smm_mean', '?')}")

    return result
