#!/usr/bin/env python3
"""
Spatial Knowledge Graph extraction from map task dialogue.

Uses GPT-4.1 to extract spatial relations from speech, builds knowledge graphs,
compares dialogue-derived graphs to ground truth map structure.

Paper 1 core module: "Spatial Knowledge Graphs from Collaborative Dialogue"

Usage:
    from knowledge_graph import process_trial, process_session

Requirements:
    pip install openai networkx numpy Pillow
"""

import base64
import hashlib
import json
import os
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ── Landmark cache (vision extraction is expensive) ──

_LANDMARK_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".landmark_cache")


def _cache_path(map_number: int, variant: str) -> str:
    os.makedirs(_LANDMARK_CACHE_DIR, exist_ok=True)
    return os.path.join(_LANDMARK_CACHE_DIR, f"landmarks_map{map_number}_{variant}.json")


def _load_cache(map_number: int, variant: str) -> Optional[List[Dict]]:
    path = _cache_path(map_number, variant)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_cache(map_number: int, variant: str, landmarks: List[Dict]):
    path = _cache_path(map_number, variant)
    with open(path, "w") as f:
        json.dump(landmarks, f, indent=2)


# ── Vision: Extract landmarks from map image ──

LANDMARK_EXTRACTION_PROMPT = """Analyze this map image from a collaborative navigation task.

Identify ALL visible landmarks on the map. For each landmark, provide:
1. **name**: A short descriptive name (e.g., "church", "oak tree", "lake", "telephone box")
2. **type**: Category — one of: building, nature, water, infrastructure, marker, other
3. **x_pct**: Approximate horizontal position as percentage (0=left, 100=right)
4. **y_pct**: Approximate vertical position as percentage (0=top, 100=bottom)
5. **size**: Relative size — small, medium, large

Also identify if there is a route/path drawn on the map.

Return JSON with:
- "landmarks": array of landmark objects
- "has_route": boolean
- "route_description": brief text description of the route path if visible"""


def extract_landmarks_from_map(map_image_path: str, map_number: int,
                                variant: str = "f", api_key: str = None,
                                use_cache: bool = True) -> List[Dict]:
    """
    Extract landmarks from a map image using GPT-4.1 vision.

    Args:
        map_image_path: Path to map GIF/PNG
        map_number: Map number (0-15)
        variant: 'f' (full/director) or 'g' (ground/matcher)
        use_cache: Cache results to avoid repeated API calls
    """
    if use_cache:
        cached = _load_cache(map_number, variant)
        if cached is not None:
            return cached

    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # Encode image as base64
    with open(map_image_path, "rb") as f:
        img_bytes = f.read()

    # Detect format from extension
    ext = os.path.splitext(map_image_path)[1].lower()
    media_type = {".gif": "image/gif", ".png": "image/png", ".jpg": "image/jpeg",
                  ".jpeg": "image/jpeg"}.get(ext, "image/gif")

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    resp = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": LANDMARK_EXTRACTION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"This is map {map_number} (variant: {'full/director' if variant == 'f' else 'ground/matcher'}). Extract all landmarks."},
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{b64}",
                    "detail": "high"
                }},
            ]},
        ],
        temperature=0.1,
        max_tokens=3000,
    )

    try:
        result = json.loads(resp.choices[0].message.content)
        landmarks = result.get("landmarks", [])
    except (json.JSONDecodeError, IndexError):
        landmarks = []

    if use_cache and landmarks:
        _save_cache(map_number, variant, landmarks)

    return landmarks


# ── Spatial Relation Extraction from Dialogue ──

SPATIAL_EXTRACTION_PROMPT = """You are analyzing a map task dialogue to extract a spatial knowledge graph.

The Director is guiding the Matcher to draw a route on a map by describing landmarks and their spatial relationships.

From the dialogue, extract ALL spatial relations mentioned or implied. For each relation:
1. **source**: The landmark or location being referenced FROM
2. **target**: The landmark or location being referenced TO
3. **relation**: The spatial relation — one of:
   - NEAR, FAR, LEFT_OF, RIGHT_OF, ABOVE, BELOW
   - NORTH_OF, SOUTH_OF, EAST_OF, WEST_OF
   - BETWEEN (use with "via" field for the third landmark)
   - ADJACENT, OPPOSITE, PAST, TOWARD, AWAY_FROM
   - ON_ROUTE, START, END, WAYPOINT
4. **direction**: If a movement direction is described (e.g., "go up", "turn right")
5. **distance**: Qualitative distance if mentioned (close, far, very far, etc.)
6. **confidence**: Your confidence in this extraction (0.0-1.0)
7. **utterance_index**: Which utterance(s) this was extracted from (approximate)

Also extract:
- **landmarks_mentioned**: List of all unique landmarks referenced in dialogue
- **route_sequence**: Ordered list of landmarks/waypoints along the described route
- **spatial_frame**: What reference frame is used? (egocentric/left-right, allocentric/cardinal, landmark-relative, mixed)
- **reference_frame_switches**: Number of times the frame switches during dialogue

Return JSON with "relations", "landmarks_mentioned", "route_sequence", "spatial_frame", "reference_frame_switches"."""


def extract_spatial_relations(transcript_director: str, transcript_matcher: str,
                              map_number: int, api_key: str = None) -> Dict[str, Any]:
    """Extract spatial knowledge graph from dialogue using GPT-4.1."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    dialogue = f"DIRECTOR: {transcript_director}\n\nMATCHER: {transcript_matcher}"

    resp = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SPATIAL_EXTRACTION_PROMPT},
            {"role": "user", "content": f"Map {map_number} dialogue:\n\n{dialogue}"},
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    try:
        result = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        result = {"relations": [], "landmarks_mentioned": [], "route_sequence": []}

    return result


# ── Graph Construction ──

def build_knowledge_graph(extraction: Dict[str, Any]) -> Optional[Any]:
    """Build a NetworkX directed graph from extracted spatial relations."""
    if not HAS_NX:
        return None

    G = nx.DiGraph()

    # Add landmark nodes
    for lm in extraction.get("landmarks_mentioned", []):
        name = lm if isinstance(lm, str) else lm.get("name", str(lm))
        G.add_node(name.lower().strip(), type="landmark")

    # Add relation edges
    for rel in extraction.get("relations", []):
        src = rel.get("source", "").lower().strip()
        tgt = rel.get("target", "").lower().strip()
        relation = rel.get("relation", "NEAR")
        if src and tgt:
            G.add_node(src, type="landmark")
            G.add_node(tgt, type="landmark")
            G.add_edge(src, tgt,
                        relation=relation,
                        direction=rel.get("direction", ""),
                        distance=rel.get("distance", ""),
                        confidence=rel.get("confidence", 1.0))

    # Add route sequence edges
    route = extraction.get("route_sequence", [])
    for i in range(len(route) - 1):
        src = route[i].lower().strip() if isinstance(route[i], str) else str(route[i]).lower().strip()
        tgt = route[i + 1].lower().strip() if isinstance(route[i + 1], str) else str(route[i + 1]).lower().strip()
        if src and tgt:
            G.add_node(src, type="waypoint")
            G.add_node(tgt, type="waypoint")
            if not G.has_edge(src, tgt):
                G.add_edge(src, tgt, relation="ON_ROUTE", route_order=i)

    return G


def graph_metrics(G) -> Dict[str, float]:
    """Compute structural metrics from the knowledge graph."""
    if not HAS_NX or G is None:
        return {}

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {"kg_n_nodes": 0, "kg_n_edges": 0}

    metrics = {
        "kg_n_nodes": n_nodes,
        "kg_n_edges": n_edges,
        "kg_density": nx.density(G),
        "kg_avg_degree": n_edges / n_nodes if n_nodes > 0 else 0,
    }

    # Relation type distribution
    relation_counts = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "UNKNOWN")
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    for rel, count in relation_counts.items():
        metrics[f"kg_rel_{rel.lower()}"] = count

    # Connectivity
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    metrics["kg_n_components"] = len(components)
    metrics["kg_largest_component"] = max(len(c) for c in components) if components else 0

    # Centrality of top nodes
    if n_nodes > 1:
        try:
            degree_cent = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            metrics["kg_max_degree_centrality"] = max(degree_cent.values())
            metrics["kg_mean_degree_centrality"] = float(np.mean(list(degree_cent.values())))
            metrics["kg_max_betweenness"] = max(betweenness.values())
            metrics["kg_mean_betweenness"] = float(np.mean(list(betweenness.values())))
        except Exception:
            pass

    # Clustering (undirected)
    try:
        metrics["kg_avg_clustering"] = nx.average_clustering(undirected)
    except Exception:
        metrics["kg_avg_clustering"] = 0.0

    # Route sequence length
    route_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "ON_ROUTE"]
    metrics["kg_route_length"] = len(route_edges)

    return metrics


# ── Ground Truth Comparison ──

def _build_gt_graph(gt_json: Dict, map_landmarks: List[Dict] = None) -> Optional[Any]:
    """Build a ground truth spatial graph from GT strokes + extracted landmarks."""
    if not HAS_NX:
        return None

    G = nx.DiGraph()

    # Add landmarks as nodes (from vision extraction)
    if map_landmarks:
        for lm in map_landmarks:
            name = lm.get("name", "").lower().strip()
            if name:
                G.add_node(name, type="landmark",
                           x_pct=lm.get("x_pct", 50),
                           y_pct=lm.get("y_pct", 50))

    # Build route from GT strokes
    strokes = gt_json.get("strokes", [])
    width = gt_json.get("image", {}).get("width", 651)
    height = gt_json.get("image", {}).get("height", 900)

    # Sample route points
    route_points = []
    for s in strokes:
        for p in s.get("points", s.get("polyline", [])):
            if isinstance(p, dict) and "x" in p:
                route_points.append((p["x"] / width * 100, p["y"] / height * 100))

    # Associate route with nearby landmarks
    if map_landmarks and route_points:
        # For each landmark, find nearest route point and compute distance
        route_arr = np.array(route_points)
        landmark_route_dist = []
        for lm in map_landmarks:
            lx = lm.get("x_pct", 50)
            ly = lm.get("y_pct", 50)
            dists = np.sqrt((route_arr[:, 0] - lx) ** 2 + (route_arr[:, 1] - ly) ** 2)
            min_dist = float(dists.min())
            nearest_idx = int(dists.argmin())
            name = lm.get("name", "").lower().strip()
            if name and min_dist < 15:  # within 15% of map dimensions
                landmark_route_dist.append((name, nearest_idx, min_dist))
                G.nodes[name]["on_route"] = True
                G.nodes[name]["route_distance_pct"] = min_dist

        # Sort by route position to get ordering
        landmark_route_dist.sort(key=lambda x: x[1])
        for i in range(len(landmark_route_dist) - 1):
            src = landmark_route_dist[i][0]
            tgt = landmark_route_dist[i + 1][0]
            G.add_edge(src, tgt, relation="ON_ROUTE", route_order=i)

    # Add spatial adjacency edges between nearby landmarks
    if map_landmarks:
        for i, lm1 in enumerate(map_landmarks):
            for lm2 in map_landmarks[i + 1:]:
                n1 = lm1.get("name", "").lower().strip()
                n2 = lm2.get("name", "").lower().strip()
                if not n1 or not n2:
                    continue
                dx = lm1.get("x_pct", 50) - lm2.get("x_pct", 50)
                dy = lm1.get("y_pct", 50) - lm2.get("y_pct", 50)
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist < 20:  # nearby
                    # Determine relation from positions
                    if abs(dx) > abs(dy):
                        rel = "RIGHT_OF" if dx > 0 else "LEFT_OF"
                    else:
                        rel = "BELOW" if dy > 0 else "ABOVE"
                    G.add_edge(n1, n2, relation=rel, distance_pct=float(dist))

    return G


def compare_to_ground_truth(dialogue_graph, gt_graph,
                            extraction: Dict = None) -> Dict[str, float]:
    """
    Compare dialogue-extracted knowledge graph to ground truth.

    Metrics:
    - Landmark recall: what fraction of GT landmarks appear in dialogue
    - Route order similarity: Kendall tau of route ordering
    - Edge overlap: shared spatial relations
    - Graph edit distance (normalized)
    """
    if not HAS_NX or dialogue_graph is None or gt_graph is None:
        return {}

    metrics = {}

    # Landmark recall / precision
    gt_nodes = set(gt_graph.nodes())
    dial_nodes = set(dialogue_graph.nodes())

    if gt_nodes:
        # Fuzzy matching: check if any GT node is a substring of any dialogue node or vice versa
        matched_gt = set()
        matched_dial = set()
        for gn in gt_nodes:
            for dn in dial_nodes:
                if gn in dn or dn in gn or _fuzzy_match(gn, dn):
                    matched_gt.add(gn)
                    matched_dial.add(dn)

        metrics["gt_landmark_recall"] = len(matched_gt) / len(gt_nodes) if gt_nodes else 0.0
        metrics["gt_landmark_precision"] = len(matched_dial) / len(dial_nodes) if dial_nodes else 0.0
        f1_num = 2 * metrics["gt_landmark_recall"] * metrics["gt_landmark_precision"]
        f1_den = metrics["gt_landmark_recall"] + metrics["gt_landmark_precision"]
        metrics["gt_landmark_f1"] = f1_num / f1_den if f1_den > 0 else 0.0
    else:
        metrics["gt_landmark_recall"] = 0.0
        metrics["gt_landmark_precision"] = 0.0
        metrics["gt_landmark_f1"] = 0.0

    # Route order comparison (Kendall tau)
    gt_route_edges = [(u, v) for u, v, d in gt_graph.edges(data=True) if d.get("relation") == "ON_ROUTE"]
    gt_route = [u for u, v in gt_route_edges]
    if gt_route_edges:
        gt_route.append(gt_route_edges[-1][1])
    dial_route_seq = extraction.get("route_sequence", []) if extraction else []
    dial_route_seq = [r.lower().strip() if isinstance(r, str) else str(r).lower().strip()
                      for r in dial_route_seq]

    if gt_route and dial_route_seq and len(gt_route) >= 2 and len(dial_route_seq) >= 2:
        # Map GT route nodes to indices
        gt_order = {node: i for i, node in enumerate(gt_route)}
        # Find common landmarks
        common = []
        for dn in dial_route_seq:
            for gn in gt_order:
                if gn in dn or dn in gn or _fuzzy_match(gn, dn):
                    common.append((gt_order[gn], len(common)))
                    break
        if len(common) >= 2:
            from scipy.stats import kendalltau
            gt_ranks = [c[0] for c in common]
            dial_ranks = [c[1] for c in common]
            tau, p_val = kendalltau(gt_ranks, dial_ranks)
            metrics["route_order_tau"] = float(tau) if np.isfinite(tau) else 0.0
            metrics["route_order_p"] = float(p_val) if np.isfinite(p_val) else 1.0

    metrics["gt_n_landmarks"] = len(gt_nodes)
    metrics["dial_n_landmarks"] = len(dial_nodes)
    metrics["gt_n_edges"] = gt_graph.number_of_edges()
    metrics["dial_n_edges"] = dialogue_graph.number_of_edges()

    # Edge relation overlap
    gt_rel_set = set()
    for u, v, d in gt_graph.edges(data=True):
        gt_rel_set.add((u, v, d.get("relation", "")))
    dial_rel_set = set()
    for u, v, d in dialogue_graph.edges(data=True):
        dial_rel_set.add((u, v, d.get("relation", "")))

    # Fuzzy edge matching (match if nodes fuzzy-match and relation matches)
    matched_edges = 0
    matched_gt_indices = set()
    gt_rel_list = list(gt_rel_set)
    for gi, (gu, gv, gr) in enumerate(gt_rel_list):
        if gi in matched_gt_indices:
            continue
        for du, dv, dr in dial_rel_set:
            if (_fuzzy_match(gu, du) and _fuzzy_match(gv, dv) and gr == dr):
                matched_edges += 1
                matched_gt_indices.add(gi)
                break

    metrics["edge_recall"] = matched_edges / len(gt_rel_set) if gt_rel_set else 0.0
    metrics["edge_precision"] = matched_edges / len(dial_rel_set) if dial_rel_set else 0.0

    return metrics


def _fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Simple fuzzy string matching using character overlap."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    if a in b or b in a:
        return True
    # Jaccard on character bigrams
    def bigrams(s):
        return set(s[i:i + 2] for i in range(len(s) - 1))
    bg_a = bigrams(a)
    bg_b = bigrams(b)
    if not bg_a or not bg_b:
        return False
    jaccard = len(bg_a & bg_b) / len(bg_a | bg_b)
    return jaccard >= threshold


# ── Temporal Evolution (windowed knowledge graph) ──

def extract_temporal_kg(segments: List[Dict], map_number: int,
                        window_size: int = 3, api_key: str = None) -> List[Dict]:
    """
    Build knowledge graphs from sliding windows of dialogue segments.

    Tracks how the spatial knowledge graph grows over time.
    Returns list of dicts with cumulative graph metrics per window.
    """
    if not segments:
        return []

    results = []
    for end_idx in range(window_size, len(segments) + 1, max(1, window_size // 2)):
        start_idx = max(0, end_idx - window_size)
        window_text = " ".join(
            seg.get("text", "") for seg in segments[start_idx:end_idx]
        )

        if not window_text.strip():
            continue

        # Use cumulative text up to this point for graph building
        cumulative_text = " ".join(seg.get("text", "") for seg in segments[:end_idx])

        try:
            extraction = extract_spatial_relations(cumulative_text, "", map_number, api_key)
            G = build_knowledge_graph(extraction)
            if G:
                m = graph_metrics(G)
                m["window_end_idx"] = end_idx
                m["window_n_segments"] = end_idx
                m["cumulative_landmarks"] = len(extraction.get("landmarks_mentioned", []))
                results.append(m)
        except Exception:
            continue

    return results


# ── Public API ──

def process_trial(transcript_director: str, transcript_matcher: str,
                  map_number: int, gt_json: Dict = None,
                  map_image_dir: str = None,
                  api_key: str = None) -> Dict[str, Any]:
    """
    Full knowledge graph pipeline for a single trial.

    Returns flat dict of features for CSV output.
    """
    result = {"map_number": map_number}

    # 1. Extract landmarks from map image (cached)
    map_landmarks = []
    if map_image_dir:
        for variant in ["f", "g"]:
            img_path = os.path.join(map_image_dir, f"map{map_number}{variant}.gif")
            if os.path.exists(img_path):
                try:
                    map_landmarks = extract_landmarks_from_map(
                        img_path, map_number, variant, api_key)
                    result[f"vision_landmarks_{variant}"] = len(map_landmarks)
                    break
                except Exception as e:
                    result[f"vision_error_{variant}"] = str(e)[:100]

    # 2. Extract spatial relations from dialogue
    if not transcript_director.strip() and not transcript_matcher.strip():
        result["kg_error"] = "empty_transcripts"
        return result

    try:
        extraction = extract_spatial_relations(
            transcript_director, transcript_matcher, map_number, api_key)
    except Exception as e:
        result["kg_error"] = str(e)[:200]
        return result

    result["n_relations_extracted"] = len(extraction.get("relations", []))
    result["n_landmarks_mentioned"] = len(extraction.get("landmarks_mentioned", []))
    result["route_sequence_length"] = len(extraction.get("route_sequence", []))
    result["spatial_frame"] = extraction.get("spatial_frame", "")
    result["reference_frame_switches"] = extraction.get("reference_frame_switches", 0)

    # 3. Build knowledge graph
    G = build_knowledge_graph(extraction)
    if G:
        kg_m = graph_metrics(G)
        result.update(kg_m)

    # 4. Compare to ground truth
    if gt_json and map_landmarks:
        gt_graph = _build_gt_graph(gt_json, map_landmarks)
        if gt_graph and G:
            gt_m = compare_to_ground_truth(G, gt_graph, extraction)
            result.update(gt_m)

    return result


def process_session(transcripts: List[Dict], gt_dir: str = None,
                    map_image_dir: str = None,
                    api_key: str = None) -> List[Dict[str, Any]]:
    """
    Process all trials in a session.

    Args:
        transcripts: list of dicts with keys:
            - trial (int)
            - map_number (int)
            - director_text (str)
            - matcher_text (str)
            - sessionId (str, optional)
        gt_dir: Path to Ground Truth Maps directory
        map_image_dir: Path to map images directory
        api_key: OpenAI API key
    """
    results = []
    for t in transcripts:
        trial = t.get("trial", 0)
        map_number = t.get("map_number", 0)

        # Load GT if available
        gt_json = None
        if gt_dir:
            gt_path = os.path.join(gt_dir, f"gt_{map_number}.json")
            if os.path.exists(gt_path):
                try:
                    with open(gt_path) as f:
                        gt_json = json.load(f)
                except Exception:
                    pass

        result = process_trial(
            t.get("director_text", ""),
            t.get("matcher_text", ""),
            map_number,
            gt_json=gt_json,
            map_image_dir=map_image_dir,
            api_key=api_key,
        )
        result["trial"] = trial
        result["sessionId"] = t.get("sessionId", "")
        results.append(result)

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract spatial knowledge graph from dialogue")
    ap.add_argument("--director-text", required=True, help="Director transcript")
    ap.add_argument("--matcher-text", required=True, help="Matcher transcript")
    ap.add_argument("--map-number", type=int, required=True, help="Map number (0-15)")
    ap.add_argument("--gt-dir", default=None, help="Ground Truth Maps directory")
    ap.add_argument("--map-image-dir", default=None, help="Map images directory")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    result = process_trial(
        args.director_text, args.matcher_text, args.map_number,
        gt_json=None,  # would need to load
        map_image_dir=args.map_image_dir,
        api_key=args.api_key,
    )
    print(json.dumps(result, indent=2, default=str))
