# Evidence Graph

## Why

1.x kept a flat list. It recorded *what* was seen but not how pieces relate, so
nothing could express "this contradicts that", "these are the same story", or
"this claim rests on that observation".

## Structure

Nodes: `observation`, `claim`, `hypothesis`, `event`, `recommendation`.
Edges: `supports`, `contradicts`, `derives_from`, `duplicates`, `explains`.

## Six score dimensions, kept separate

| Dimension | Question |
|---|---|
| `factual_reliability` | is it true? |
| `predictive_utility` | does this *kind* of signal improve forecasts? |
| `timeliness` | how fresh, relative to its useful life? |
| `novelty` | is it new, or already priced? |
| `market_impact` | does this class of event move the name? |
| `independence` | computed by the graph, never supplied by the source |

They collapse only at the point of use, via `composite(weights)`, where the
caller chooses the weighting. **`None` means unknown and never becomes a
fabricated 0.5** — `test_unknown_score_stays_unknown`.

SEC EDGAR scores ~0.98 factual and ~0.35 predictive. Conflating those was the
1.x mistake.

## Correlation clusters

```
market_beta 0.85 · macro 0.70 · volatility 0.65 · sector 0.60 · technical 0.60
options_positioning 0.55 · sentiment 0.50 · company_news 0.45
company_fundamental 0.35 · insider 0.30
```

`effective_independent_count()` applies a two-level variance-inflation
correction: within cluster by its ρ, then across clusters by a residual 0.15 —
because on a risk-off day even different categories move together.

Twenty market-beta reads collapse to **under 2.5** effective observations.
Tested in `test_twenty_copies_of_one_signal_do_not_become_confidence`.

## Explaining a conclusion

`graph.explain(node)` answers both required questions: *why do we believe this?*
and *what would invalidate it?* When no contradicting evidence was gathered it
says so explicitly — absence of counter-evidence is not evidence of absence.
