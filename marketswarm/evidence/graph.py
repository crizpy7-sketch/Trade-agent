"""The Evidence Graph.

1.x collected evidence in a flat list. That records *what* was seen but not how
the pieces relate, so nothing could express "this contradicts that", "these two
headlines are the same story", or "this claim rests on that observation".

The graph fixes three specific failures:

1. **Untraceable conclusions.** Every recommendation can now walk its own
   support and opposition, and answer "what would invalidate this?" from
   structure rather than from a hand-written sentence.

2. **One conflated trust score.** `reliability` mixed "is this true" with "does
   this predict anything". SEC EDGAR is ~0.98 on the first and mediocre on the
   second. Six dimensions are kept separate (see `EvidenceScore`) and only
   collapsed at the point of use, where the caller chooses the weighting.

3. **Double-counted signals.** Futures, index breadth and momentum are one
   observation wearing three hats. Nodes carry a `cluster`, and effective
   independence is computed from cluster membership rather than from a single
   global correlation constant.

Nothing here fabricates evidence. A claim with no source keeps
`factual_reliability = None`, and unknown stays unknown.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable


class NodeType(str, Enum):
    OBSERVATION = "observation"      # a measured fact: SPY gapped +0.4%
    CLAIM = "claim"                  # an assertion derived from observations
    HYPOTHESIS = "hypothesis"        # a candidate explanation under test
    EVENT = "event"                  # something that happened: earnings, CPI
    RECOMMENDATION = "recommendation"


class Relation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVES_FROM = "derives_from"    # B was computed from A
    DUPLICATES = "duplicates"        # same underlying information
    EXPLAINS = "explains"


# Correlation families. Two nodes in the same cluster are treated as largely
# the same observation; the intra-cluster correlation is what the confidence
# maths discounts against.
CLUSTERS: dict[str, float] = {
    "market_beta": 0.85,      # futures, index ETFs, breadth, index momentum
    "sector": 0.60,
    "company_fundamental": 0.35,
    "company_news": 0.45,
    "macro": 0.70,
    "volatility": 0.65,
    "options_positioning": 0.55,
    "technical": 0.60,
    "sentiment": 0.50,
    "insider": 0.30,
    "unknown": 0.50,
}

DEFAULT_CROSS_CLUSTER_RHO = 0.15
"""Residual correlation between different clusters. Not zero: on a risk-off day
almost everything moves together, so pretending clusters are independent would
reintroduce the overconfidence this module exists to prevent."""


@dataclass
class EvidenceScore:
    """Six dimensions, deliberately not collapsed.

    Each is in [0, 1], or None when genuinely unknown — which is not the same
    as zero and must not be silently treated as such.

    factual_reliability : is the claim likely true?
    predictive_utility  : has this *kind* of signal historically improved
                          forecasts? Learned from scored history; low by default.
    timeliness          : how fresh, relative to its useful life.
    novelty             : is this new, or already priced/known?
    market_impact       : does this class of event historically move the name?
    independence        : computed by the graph, not supplied by the source.
    """

    factual_reliability: float | None = None
    predictive_utility: float | None = None
    timeliness: float | None = None
    novelty: float | None = None
    market_impact: float | None = None
    independence: float | None = None

    def known(self) -> dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def composite(self, weights: dict[str, float] | None = None) -> float | None:
        """Collapse to one number *at the point of use*.

        Returns None when nothing is known — callers must handle that rather
        than receive a fabricated 0.5.
        """
        known = self.known()
        if not known:
            return None
        w = weights or {
            "factual_reliability": 0.30,
            "predictive_utility": 0.30,
            "timeliness": 0.15,
            "novelty": 0.10,
            "market_impact": 0.10,
            "independence": 0.05,
        }
        num = sum(known[k] * w.get(k, 0.0) for k in known)
        den = sum(w.get(k, 0.0) for k in known)
        return float(num / den) if den > 0 else None

    def weakest(self) -> tuple[str, float] | None:
        known = self.known()
        if not known:
            return None
        k = min(known, key=known.get)
        return k, known[k]


@dataclass
class EvidenceNode:
    claim: str
    node_type: NodeType = NodeType.OBSERVATION
    subject: str | None = None
    source: str | None = None
    source_type: str | None = None
    url: str | None = None
    raw_observation: str | None = None
    observed_at: str | None = None
    retrieved_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    cluster: str = "unknown"
    score: EvidenceScore = field(default_factory=EvidenceScore)
    is_inferred: bool = False
    provenance: str | None = None
    tags: list[str] = field(default_factory=list)
    agent: str | None = None
    id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")

    @property
    def age_hours(self) -> float | None:
        if not self.observed_at:
            return None
        try:
            t = dt.datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return (dt.datetime.now(dt.timezone.utc) - t).total_seconds() / 3600

    def describe(self) -> str:
        bits = [self.claim]
        if self.source:
            bits.append(f"[{self.source}]")
        if self.is_inferred:
            bits.append("(inferred, not observed)")
        return " ".join(bits)

    def to_row(self, investigation_id: str | None) -> tuple:
        import json
        return (
            self.id, investigation_id, self.retrieved_at, self.node_type.value,
            self.subject, self.claim, self.raw_observation, self.source,
            self.source_type, self.url, self.observed_at, self.retrieved_at,
            self.cluster, self.score.factual_reliability, self.score.predictive_utility,
            self.score.timeliness, self.score.novelty, self.score.market_impact,
            self.score.independence, self.score.composite(),
            1 if self.is_inferred else 0, self.provenance, json.dumps(self.tags),
        )


@dataclass
class EvidenceEdge:
    src_id: str
    dst_id: str
    relation: Relation
    weight: float = 1.0
    note: str = ""
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class EvidenceGraph:
    """A directed multigraph of evidence for one investigation."""

    def __init__(self, investigation_id: str | None = None):
        self.investigation_id = investigation_id
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: list[EvidenceEdge] = []

    # ---------- construction ----------

    def add(self, node: EvidenceNode) -> EvidenceNode:
        self.nodes[node.id] = node
        return node

    def observe(self, claim: str, **kw) -> EvidenceNode:
        return self.add(EvidenceNode(claim=claim, node_type=NodeType.OBSERVATION, **kw))

    def hypothesize(self, claim: str, **kw) -> EvidenceNode:
        return self.add(EvidenceNode(claim=claim, node_type=NodeType.HYPOTHESIS, **kw))

    def link(self, src: EvidenceNode | str, dst: EvidenceNode | str,
             relation: Relation, weight: float = 1.0, note: str = "") -> EvidenceEdge:
        s = src.id if isinstance(src, EvidenceNode) else src
        d = dst.id if isinstance(dst, EvidenceNode) else dst
        if s not in self.nodes or d not in self.nodes:
            raise KeyError("both endpoints must be in the graph before linking")
        e = EvidenceEdge(s, d, relation, weight, note)
        self.edges.append(e)
        return e

    # ---------- queries ----------

    def by_type(self, node_type: NodeType) -> list[EvidenceNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def by_subject(self, subject: str) -> list[EvidenceNode]:
        s = subject.upper()
        return [n for n in self.nodes.values() if (n.subject or "").upper() == s]

    def supporting(self, node: EvidenceNode | str) -> list[EvidenceNode]:
        nid = node.id if isinstance(node, EvidenceNode) else node
        return [self.nodes[e.src_id] for e in self.edges
                if e.dst_id == nid and e.relation == Relation.SUPPORTS
                and e.src_id in self.nodes]

    def contradicting(self, node: EvidenceNode | str) -> list[EvidenceNode]:
        nid = node.id if isinstance(node, EvidenceNode) else node
        out = []
        for e in self.edges:
            if e.relation != Relation.CONTRADICTS:
                continue
            if e.dst_id == nid and e.src_id in self.nodes:
                out.append(self.nodes[e.src_id])
            elif e.src_id == nid and e.dst_id in self.nodes:
                out.append(self.nodes[e.dst_id])   # contradiction is symmetric
        return out

    def contradictions(self) -> list[tuple[EvidenceNode, EvidenceNode]]:
        return [
            (self.nodes[e.src_id], self.nodes[e.dst_id])
            for e in self.edges
            if e.relation == Relation.CONTRADICTS
            and e.src_id in self.nodes and e.dst_id in self.nodes
        ]

    def provenance_chain(self, node: EvidenceNode | str, depth: int = 6) -> list[EvidenceNode]:
        """Walk `derives_from` edges back to the raw observations."""
        nid = node.id if isinstance(node, EvidenceNode) else node
        chain: list[EvidenceNode] = []
        seen = {nid}
        frontier = [nid]
        for _ in range(depth):
            nxt = []
            for cur in frontier:
                for e in self.edges:
                    if e.src_id == cur and e.relation == Relation.DERIVES_FROM:
                        if e.dst_id in self.nodes and e.dst_id not in seen:
                            seen.add(e.dst_id)
                            chain.append(self.nodes[e.dst_id])
                            nxt.append(e.dst_id)
            if not nxt:
                break
            frontier = nxt
        return chain

    # ---------- independence ----------

    def cluster_counts(self, nodes: Iterable[EvidenceNode] | None = None) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for n in (nodes if nodes is not None else self.nodes.values()):
            counts[n.cluster] += 1
        return dict(counts)

    def effective_independent_count(
        self, nodes: Iterable[EvidenceNode] | None = None,
        cross_cluster_rho: float = DEFAULT_CROSS_CLUSTER_RHO,
    ) -> float:
        """How many genuinely independent observations do we actually have?

        Two-level correction. Within a cluster, k items with correlation rho
        count as k / (1 + rho(k-1)) — the standard variance-inflation result for
        exchangeable correlation. Across clusters, the same correction is applied
        again with a small residual rho, because on a risk-off day even
        "independent" categories move together.

        Ten bullish momentum reads and one SEC filing is not eleven pieces of
        evidence, and this is the function that says so.
        """
        pool = list(nodes if nodes is not None else self.nodes.values())
        if not pool:
            return 0.0

        per_cluster: list[float] = []
        for cluster, items in _group_by_cluster(pool).items():
            k = len(items)
            rho = CLUSTERS.get(cluster, CLUSTERS["unknown"])
            per_cluster.append(k / (1.0 + rho * (k - 1)) if k > 1 else float(k))

        m = len(per_cluster)
        total = sum(per_cluster)
        if m <= 1:
            return float(total)
        rho_x = max(0.0, min(0.95, cross_cluster_rho))
        return float(total / (1.0 + rho_x * (m - 1)))

    def independence_of(self, node: EvidenceNode) -> float:
        """A node's own independence: 1.0 when it is the only member of its
        cluster, decaying as the cluster fills up with near-duplicates."""
        same = [n for n in self.nodes.values() if n.cluster == node.cluster]
        k = len(same)
        if k <= 1:
            return 1.0
        rho = CLUSTERS.get(node.cluster, CLUSTERS["unknown"])
        return float(1.0 / (1.0 + rho * (k - 1)))

    def annotate_independence(self) -> None:
        """Fill in the independence dimension for every node. Cheap, and it is
        the one score the source cannot supply for itself."""
        for n in self.nodes.values():
            n.score.independence = self.independence_of(n)

    # ---------- summary ----------

    def summary(self) -> dict:
        nodes = list(self.nodes.values())
        inferred = [n for n in nodes if n.is_inferred]
        scored = [n for n in nodes if n.score.composite() is not None]
        contradictions = self.contradictions()
        return {
            "n_nodes": len(nodes),
            "n_edges": len(self.edges),
            "by_type": {t.value: len(self.by_type(t)) for t in NodeType
                        if self.by_type(t)},
            "clusters": self.cluster_counts(),
            "effective_independent": round(self.effective_independent_count(), 2),
            "n_inferred": len(inferred),
            "n_contradictions": len(contradictions),
            "mean_composite": (
                round(sum(n.score.composite() for n in scored) / len(scored), 3)
                if scored else None
            ),
            "unscored": len(nodes) - len(scored),
        }

    def explain(self, node: EvidenceNode | str, max_items: int = 6) -> dict:
        """Why do we believe this, and what would break it?"""
        n = self.nodes[node] if isinstance(node, str) else node
        sup = sorted(self.supporting(n),
                     key=lambda x: -(x.score.composite() or 0))[:max_items]
        con = sorted(self.contradicting(n),
                     key=lambda x: -(x.score.composite() or 0))[:max_items]
        return {
            "claim": n.claim,
            "believe_because": [s.describe() for s in sup],
            "would_be_invalidated_by": (
                [c.describe() for c in con]
                or ["no contradicting evidence was gathered — absence of "
                    "counter-evidence is not evidence of absence"]
            ),
            "support_clusters": sorted({s.cluster for s in sup}),
            "effective_independent_support": round(
                self.effective_independent_count(sup), 2),
            "provenance": [p.describe() for p in self.provenance_chain(n)],
        }


def _group_by_cluster(nodes: list[EvidenceNode]) -> dict[str, list[EvidenceNode]]:
    out: dict[str, list[EvidenceNode]] = defaultdict(list)
    for n in nodes:
        out[n.cluster].append(n)
    return dict(out)


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------

def timeliness_from_age(age_hours: float | None, half_life_hours: float = 12.0) -> float | None:
    """Exponential decay. A headline is worth much less 20 hours later; a 10-K
    is not. Callers pass the appropriate half-life."""
    if age_hours is None:
        return None
    if age_hours < 0:
        age_hours = 0.0
    return float(math.exp(-math.log(2) * age_hours / max(half_life_hours, 1e-6)))


def novelty_from_corroboration(corroborations: int, already_known: bool = False) -> float:
    """Widely repeated news is usually already priced.

    Deliberately inverted relative to factual reliability: more outlets means
    *more* likely true and *less* likely to still be tradeable.
    """
    if already_known:
        return 0.1
    if corroborations <= 1:
        return 0.9
    return float(max(0.15, 0.9 / corroborations))


# Cold-start predictive utility by source class. Low across the board on
# purpose: nothing here has demonstrated predictive value yet, and these are
# replaced by measured values as scored history accumulates.
DEFAULT_PREDICTIVE_UTILITY: dict[str, float] = {
    "sec_edgar": 0.35,
    "exchange_data": 0.30,
    "company_ir": 0.30,
    "cboe": 0.30,
    "federal_reserve": 0.25,
    "treasury": 0.20,
    "bls": 0.20,
    "major_newswire": 0.22,
    "financial_media": 0.15,
    "analyst_estimate": 0.12,
    "aggregator": 0.10,
    "social_sentiment": 0.05,
    "unknown": 0.10,
}


def default_predictive_utility(source_type: str | None) -> float:
    return DEFAULT_PREDICTIVE_UTILITY.get(source_type or "unknown",
                                          DEFAULT_PREDICTIVE_UTILITY["unknown"])
