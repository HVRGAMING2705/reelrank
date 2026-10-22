"""Builds the ReelRank project presentation (9 slides).
Real eval numbers are read from movies.json so the results slide shows the
actual genre-agreement P@5 and random baseline.
"""
import json

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls


def set_cell_borders(cell, color="000000", w="12700"):  # w in EMU; 12700 = 1pt
    """Give a table cell plain black grid borders via raw XML
    (python-pptx exposes no public border API)."""
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnT", "a:lnL", "a:lnB", "a:lnR"):
        ln = parse_xml(
            f'<{tag} {nsdecls("a")} w="{w}">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></{tag}>'
        )
        tcPr.append(ln)

with open("movies.json") as f:
    MJ = json.load(f)
EV = MJ["eval"]
P5 = f"{EV['genre_agreement_p5'] * 100:.1f}%"
BASE = f"{EV['random_baseline_p5'] * 100:.1f}%"
LIFT = f"{EV['genre_agreement_p5'] / EV['random_baseline_p5']:.1f}x"
DIM = EV["feature_dim"]
N = EV["n_movies"]

BLACK = RGBColor(0x00, 0x00, 0x00)
GREY = RGBColor(0x44, 0x44, 0x44)
FONT = "Times New Roman"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def txbox(slide, left, top, width, height, text, size=20, bold=False,
          color=BLACK, align=None, font=FONT):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font
    if align is not None:
        p.alignment = align
    return tb


def number_tag(slide, n):
    txbox(slide, 0.8, 0.35, 2, 0.6, f"{n:02d}", size=18, bold=True, color=GREY)


def content_slide(title, n, body_lines, table_rows=None):
    """Title + body text lines, optional plain black grid table under the text."""
    s = prs.slides.add_slide(blank)
    number_tag(s, n)
    txbox(s, 0.8, 0.85, 11.7, 1.1, title, size=36, bold=True)
    y = 2.05
    for line in body_lines:
        txbox(s, 0.8, y, 11.7, 0.6, line, size=20)
        y += 0.55
    if table_rows:
        rows, cols = len(table_rows), len(table_rows[0])
        g = s.shapes.add_table(rows, cols, Inches(0.8), Inches(y + 0.25),
                               Inches(8.5), Inches(0.45 * rows))
        tbl = g.table
        for r in range(rows):
            for c in range(cols):
                cell = tbl.cell(r, c)
                cell.text = table_rows[r][c]
                for p in cell.text_frame.paragraphs:
                    p.font.size = Pt(16)
                    p.font.name = FONT
                    p.font.color.rgb = BLACK
                    p.font.bold = (r == 0)
                set_cell_borders(cell, "000000", "12700")
        for r in range(rows):
            tbl.rows[r].height = Inches(0.45)
    return s


# 1 — Title
s = prs.slides.add_slide(blank)
txbox(s, 1.0, 1.6, 11.3, 1.4, "ReelRank", size=56, bold=True, align=PP_ALIGN.CENTER)
txbox(s, 1.0, 3.1, 11.3, 1.0, "AI Movie Recommender", size=28,
      align=PP_ALIGN.CENTER)
txbox(s, 1.0, 4.4, 11.3, 2.0,
      "Vikranth Reddy  |  B.Tech CSE (AI & ML), Woxsen University\n"
      "Daily AI/ML Project Factory — Day 5  ·  22 October 2026",
      size=20, color=GREY, align=PP_ALIGN.CENTER)

# 2 — Problem
content_slide("The Problem", 2, [
    "•  Too much choice: streaming catalogs have thousands of titles, and most",
    "    people re-watch the same few movies instead of discovering new ones.",
    "•  Simple genre filters are too coarse: two 'Sci-Fi' movies can feel",
    "    completely different (a space epic vs. a robot comedy).",
    "•  Big platforms use collaborative filtering — but that needs millions of",
    "    users and fails for new items (the cold-start problem).",
    "•  Goal: pick 1–3 movies you love → get ranked, explainable recommendations",
    "    from the movies' own content, with no user data at all.",
])

# 3 — Solution
content_slide("The Solution", 3, [
    "•  ReelRank: a content-based recommender over a catalog of 64 synthetic",
    f"    movies (1990–2026), each encoded as a {DIM}-dim feature vector — genre",
    "    one-hot (weight ×2), keyword one-hot, era bin and runtime bin, L2-normalized.",
    "•  Your 1–3 picks are averaged into a 'taste profile'; the other 63 movies",
    "    are ranked by cosine similarity (dot product on unit vectors).",
    "•  The catalog and feature matrix ship to the browser as movies.json — the",
    "    website re-runs the exact same math in JavaScript. Zero backend.",
    "•  Each recommendation shows a match-% bar and the shared genres as the reason.",
])

# 4 — Tech Stack
content_slide("Tech Stack", 4, [
    "•  Python, NumPy — catalog, feature engineering and evaluation (train.py).",
    f"•  Content-based filtering: {DIM}-dim movie vectors (12 genres ×2 + keyword",
    "    vocabulary + 4 era bins + 3 runtime bins), L2-normalized; query = mean of",
    "    picked movies; ranking by cosine similarity.",
    "•  Vanilla JavaScript — search/select up to 3 favorites, count slider (3–12),",
    "    top-N cards with match bars and shared-genre explanations.",
    "•  Static hosting on Vercel with hardened security headers.",
])

# 5 — Architecture
content_slide("Architecture", 5, [
    "•  train.py: (1) hardcode the 64-movie catalog → (2) build keyword/genre/era/",
    "    runtime one-hot features → (3) L2-normalize → (4) evaluate: leave-one-movie-",
    "    out genre-agreement precision@5 over all 64 movies + a 10-seed random",
    "    baseline → (5) export movies.json.",
    f"•  movies.json: {N} movies + {N} × {DIM} feature matrix (~81 KB) — small",
    "    enough to ship to the browser.",
    "•  index.html: fetch movies.json → search box adds picks as removable chips →",
    "    JS builds the taste profile → cosine similarity → top-N cards with match-",
    "    % bars and shared-genre explanations + a 'Surprise me' random button.",
    "•  Node parity check: the same cosine math run in JS matches Python to 1e-9.",
])

# 6 — Demo
content_slide("Live Demo", 6, [
    "•  Type a movie name, click it to add it as a chip (1–3 picks max).",
    "•  Move the slider to choose 3–12 recommendations, hit 'Get recommendations' —",
    "    cards show match % bars and which genres each shares with your picks.",
    "•  'Surprise me' picks random favorites and still explains every recommendation.",
    "•  Demo URL: reelrank.vercel.app (placeholder domain — updating after deploy)",
])

# 7 — Results (real eval numbers from movies.json)
eval_rows = [["Metric", "ReelRank", "Random baseline"],
             ["Genre-agreement P@5", P5, BASE],
             ["Lift over random", LIFT, "—"],
             ["Movies evaluated", str(N), str(N)]]
content_slide("Results", 7, [
    f"•  Mean genre-agreement precision@5: {P5} over all {N} movies (each used as",
    "    a single-pick query; fraction of top-5 sharing ≥ 1 genre).",
    f"•  Random baseline: {BASE} (10 seeds) → ReelRank gives a {LIFT} lift.",
    "•  JS/browser recommendations match Python's cosine math to 1e-9 — the demo",
    "    is not an approximation, it IS the model.",
    "•  Known limitation: genre weight ×2 dominates, so recommendations cluster",
    "    tightly by genre — great for precision, weaker for cross-genre discovery.",
], table_rows=eval_rows)

# 8 — Future Scope
content_slide("Future Scope", 8, [
    "•  Add ratings and real user profiles → hybrid collaborative + content filtering.",
    "•  Learn feature weights from click data instead of the fixed genre ×2 weight.",
    "•  Diversity-aware re-ranking (MMR) so all 5 picks aren't the same sub-genre.",
    "•  Scale to a real dataset (e.g. MovieLens) and compare against matrix factorization.",
    "•  Explanations from keyword overlap, not just shared genres.",
])

# 9 — Thank You
s = prs.slides.add_slide(blank)
txbox(s, 1.5, 2.2, 10.3, 1.6, "Thank You", size=60, bold=True, align=PP_ALIGN.CENTER)
txbox(s, 1.5, 4.3, 10.3, 1.5,
      "GitHub: github.com/HVRGAMING2705/reelrank\nDemo: reelrank.vercel.app",
      size=22, color=GREY, align=PP_ALIGN.CENTER)

out = "/home/hatch/workspace/goals/daily-ai-ml-project-factory/files/AIML_ReelRank_2026-10-22.pptx"
prs.save(out)
print("saved", out)
print(f"slides: {len(prs.slides)} | P@5: {P5} | random baseline: {BASE} | lift: {LIFT}")
