# ReelRank 🎬

A **content-based movie recommendation engine** — pick 1–3 favorite movies and a real cosine-similarity model ranks what you should watch next, right in the browser.

**Live demo:** https://reelrank.vercel.app (placeholder domain — to be fixed to the real Vercel domain after deploy)

## How it works

1. **Catalog** — 64 synthetic movies (1990–2026), each with 1–2 genres from a fixed list of 12, 3–6 keyword tags, an era bin (90s/2000s/2010s/2020s) and a runtime bin (<100 / 100–140 / >140 min).
2. **Features** — Each movie becomes a 216-dim vector: genre one-hot (weight ×2), keyword one-hot, era one-hot, runtime one-hot — then L2-normalized.
3. **Query** — Your 1–3 picks are averaged into a "taste profile" vector (also normalized). Recommendations = the other 63 movies ranked by cosine similarity (plain dot product on unit vectors).
4. **Evaluation** — Leave-one-movie-out: each of the 64 movies used as a single-pick query. Mean **genre-agreement precision@5: 100.0%** (fraction of top-5 recommendations sharing ≥ 1 genre with the query), vs **27.8%** for a random recommender (10 seeds) — a **3.6× lift**.
5. **Browser demo** — `train.py` exports the catalog + feature matrix to `movies.json`; JavaScript re-runs the exact same cosine math as NumPy. No backend, no API calls. Verified: JS recommendations match Python to 1e-9 (parity **PASS**).

## Reproduce

```bash
pip install numpy
python train.py   # builds catalog, runs eval, exports movies.json, runs node parity check
```

## Files

- `train.py` — catalog, feature engineering, evaluation, movies.json export (+ JS parity check)
- `movies.json` — 64 movies + 216-dim feature matrix + eval numbers (81 KB)
- `index.html` — the interactive demo site (search picks, chips, count slider, match bars)
- `make_ppt.py` — builds the project presentation (python-pptx)
- `vercel.json` — static hosting config + security headers
- `404.html`, `robots.txt`, `sitemap.xml` — standard site files

Built by [Vikranth Reddy](https://github.com/HVRGAMING2705) · B.Tech CSE (AI & ML), Woxsen University
