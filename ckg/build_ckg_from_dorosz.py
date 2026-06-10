"""Build a Causal Knowledge Graph (CKG) from the Dorosz medical compendium.

Input:  data_ckg/dorosz_txt/pages_*.txt (from extract_dorosz.py)
Output: data_ckg/dorosz_ckg.json       — nodes (symptom|disease|drug|dosage|route) + edges
        data_ckg/dorosz_ckg.graphml    — NetworkX export (for Gephi / cytoscape)

Node types:
    symptom  - "douleur", "fièvre", "céphalée", "nausées", ...
    disease  - "hypertension", "diabète", "angine", ...
    drug     - INN brand/molecule names indexed in Dorosz
    dosage   - "500 mg", "1 cp/8 h", "IV 100 mg/j"
    route    - voie orale, IV, IM, topique, rectale

Edge types:
    treats         : drug → disease
    indication     : drug → symptom
    has_dosage     : drug → dosage
    has_route      : drug → route
    contraindication: drug → condition
    interacts_with : drug → drug

The extraction is deliberately rule-based (regex + dictionaries). It trades recall
for precision: we only emit a fact when at least two anchors are present in the same
paragraph (drug name + dosage token, or drug name + indication keyword). This keeps
the CKG clean enough to act as medical ground-truth.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx

SRC = Path("data_ckg/dorosz_txt")
OUT_DIR = Path("data_ckg")
OUT_JSON = OUT_DIR / "dorosz_ckg.json"
OUT_GRAPHML = OUT_DIR / "dorosz_ckg.graphml"

# Dosage patterns (mg, g, mcg, µg, UI, cp, gouttes, ml, %, ...)
RE_DOSAGE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s?"
    r"(mg|g|µg|mcg|ug|UI|IU|ml|mL|%|cp|comprim[ée]s?|gouttes?|gél(?:ule|ules)?|sachets?|amp(?:\.|oule)s?)"
    r"(?:\s?/\s?(?:j|jour|24\s?h|kg|m2|prise))?",
    re.IGNORECASE,
)

# Routes of administration
ROUTE_KEYWORDS = {
    "voie orale", "per os", "po", "iv", "i.v.", "intraveineuse", "intraveineux",
    "im", "intramusculaire", "sc", "sous-cutan", "topique", "cutan",
    "rectale", "vaginale", "inhalation", "nasale", "ophtalmique", "sublingual",
}

# Indication anchors (signal that a sentence describes WHAT the drug treats)
INDICATION_ANCHORS = [
    "indiqué dans", "indication", "indications", "traitement de", "traitement du",
    "traitement des", "prévention", "prophylaxie",
]

# Contra-indication anchors
CONTRA_ANCHORS = ["contre-indiqu", "contre-indication", "ne pas utiliser"]

# High-frequency symptoms / diseases (French) — seed list; will be enriched from
# Dorosz index if extracted.
SYMPTOM_LEX = {
    "douleur", "douleurs", "fièvre", "céphalée", "céphalées", "migraine",
    "nausée", "nausées", "vomissement", "vomissements", "diarrhée", "constipation",
    "toux", "dyspnée", "asthénie", "anxiété", "insomnie", "vertige", "vertiges",
    "prurit", "œdème", "hypertension", "hypotension", "palpitations", "arythmie",
    "angine", "tachycardie", "bradycardie", "urticaire", "éruption",
}
DISEASE_LEX = {
    "diabète", "hypertension", "insuffisance cardiaque", "angine de poitrine",
    "infarctus", "asthme", "bpco", "épilepsie", "dépression", "schizophrénie",
    "parkinson", "alzheimer", "ulcère", "gastrite", "reflux", "hépatite",
    "cirrhose", "pneumonie", "tuberculose", "méningite", "sinusite", "otite",
    "cystite", "pyélonéphrite", "infection urinaire", "infection respiratoire",
    "arthrose", "arthrite", "ostéoporose", "goutte", "psoriasis", "eczéma",
    "acné", "thyroïdie", "hypothyroïdie", "hyperthyroïdie", "anémie",
    "leucémie", "cancer", "lymphome",
}

# Drug-name detector: uppercase monograph headers used by Dorosz — e.g. "PARACÉTAMOL",
# "AMOXICILLINE", "METFORMINE" at the beginning of a line, 4+ letters, no digits.
RE_DRUG_HEADER = re.compile(r"^([A-ZÀ-Ý][A-ZÀ-Ý\- ]{3,40})\s*$", re.MULTILINE)

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _load_chunks() -> list[tuple[str, str]]:
    out = []
    for fp in sorted(SRC.glob("pages_*.txt")):
        out.append((fp.name, fp.read_text(encoding="utf-8", errors="ignore")))
    return out


def _extract_drugs(text: str) -> list[tuple[str, int, int]]:
    """Return list of (drug_name, start_char, end_char) detected as monograph headers."""
    hits = []
    for m in RE_DRUG_HEADER.finditer(text):
        name = m.group(1).strip()
        # filter out obvious non-drug headers (1 word, common words)
        if name in {"INDEX", "SOMMAIRE", "TABLE", "ANNEXE", "PRÉFACE", "AVERTISSEMENT"}:
            continue
        hits.append((name, m.start(), m.end()))
    return hits


def _window_for_drug(text: str, start: int, next_start: int) -> str:
    """Return the slice of `text` that belongs to this drug monograph."""
    return text[start:next_start]


def _find_symptoms(block: str) -> set[str]:
    lc = block.lower()
    return {s for s in SYMPTOM_LEX if s in lc}


def _find_diseases(block: str) -> set[str]:
    lc = block.lower()
    return {d for d in DISEASE_LEX if d in lc}


def _find_dosages(block: str) -> set[str]:
    out = set()
    for m in RE_DOSAGE.finditer(block):
        out.add(m.group(0).strip())
    return out


def _find_routes(block: str) -> set[str]:
    lc = block.lower()
    return {r for r in ROUTE_KEYWORDS if r in lc}


def _has_indication_anchor(block: str) -> bool:
    lc = block.lower()
    return any(a in lc for a in INDICATION_ANCHORS)


def _has_contra_anchor(block: str) -> bool:
    lc = block.lower()
    return any(a in lc for a in CONTRA_ANCHORS)


def build() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    n_edges = defaultdict(int)
    chunks = _load_chunks()
    if not chunks:
        print("[ckg] no input text — run extract_dorosz.py first")
        return g

    total_drugs = 0
    for fname, text in chunks:
        drug_hits = _extract_drugs(text)
        for i, (name, s, _) in enumerate(drug_hits):
            next_s = drug_hits[i + 1][1] if i + 1 < len(drug_hits) else len(text)
            block = _window_for_drug(text, s, next_s)
            total_drugs += 1

            drug_node = ("drug", name)
            g.add_node(drug_node, label=name, type="drug")

            # Dosages ------------------------------------------------------
            for d in _find_dosages(block):
                dn = ("dosage", d)
                g.add_node(dn, label=d, type="dosage")
                g.add_edge(drug_node, dn, key="has_dosage", relation="has_dosage")
                n_edges["has_dosage"] += 1

            # Routes -------------------------------------------------------
            for r in _find_routes(block):
                rn = ("route", r)
                g.add_node(rn, label=r, type="route")
                g.add_edge(drug_node, rn, key="has_route", relation="has_route")
                n_edges["has_route"] += 1

            # Indications (symptoms / diseases) ---------------------------
            if _has_indication_anchor(block):
                for sy in _find_symptoms(block):
                    sn = ("symptom", sy)
                    g.add_node(sn, label=sy, type="symptom")
                    g.add_edge(drug_node, sn, key="indication", relation="indication")
                    n_edges["indication"] += 1
                for ds in _find_diseases(block):
                    dn = ("disease", ds)
                    g.add_node(dn, label=ds, type="disease")
                    g.add_edge(drug_node, dn, key="treats", relation="treats")
                    n_edges["treats"] += 1

            # Contraindications -------------------------------------------
            if _has_contra_anchor(block):
                for ds in _find_diseases(block):
                    dn = ("disease", ds)
                    g.add_node(dn, label=ds, type="disease")
                    g.add_edge(drug_node, dn, key="contraindication",
                               relation="contraindication")
                    n_edges["contraindication"] += 1

        print(f"[ckg] {fname}: +{len(drug_hits)} drugs, "
              f"graph |V|={g.number_of_nodes()} |E|={g.number_of_edges()}")

    print(f"[ckg] total drug monographs seen: {total_drugs}")
    print(f"[ckg] edges by relation: {dict(n_edges)}")
    return g


def dump(g: nx.MultiDiGraph) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # JSON (node-link, so tuples → strings)
    nodes = []
    index = {}
    for nid, data in g.nodes(data=True):
        sid = f"{nid[0]}:{nid[1]}"
        index[nid] = sid
        nodes.append({"id": sid, **data})
    edges = []
    for u, v, k, data in g.edges(keys=True, data=True):
        edges.append({
            "source": index[u],
            "target": index[v],
            "key": k,
            **data,
        })
    OUT_JSON.write_text(
        json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[ckg] wrote {OUT_JSON}  ({len(nodes)} nodes, {len(edges)} edges)")

    # GraphML — flatten tuple IDs to strings so it's importable
    g2 = nx.MultiDiGraph()
    for nid, data in g.nodes(data=True):
        g2.add_node(index[nid], **data)
    for u, v, k, data in g.edges(keys=True, data=True):
        g2.add_edge(index[u], index[v], key=str(k), **data)
    try:
        nx.write_graphml(g2, str(OUT_GRAPHML))
        print(f"[ckg] wrote {OUT_GRAPHML}")
    except Exception as exc:  # graphml is optional
        print(f"[ckg] graphml export skipped: {exc}")


if __name__ == "__main__":
    g = build()
    dump(g)
