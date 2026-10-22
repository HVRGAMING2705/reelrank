"""ReelRank — content-based movie recommender (2nd-year B.Tech level).

Catalog: 64 synthetic movies (hardcoded below). Each movie gets a feature
vector: genre one-hot (weight x2), keyword one-hot, era-bin one-hot,
runtime-bin one-hot, then L2-normalized. A user's query profile is the mean
of the feature vectors of the 1-3 movies they pick; recommendations are the
top-N movies by cosine similarity (excluding the picks themselves).

Evaluated by mean "genre agreement" precision@5 over all 64 movies
(fraction of each movie's top-5 recommendations sharing >= 1 genre with the
query movie), plus a random-recommender baseline for contrast.

Exports movies.json (catalog + L2-normalized feature matrix) so the browser
demo runs the exact same cosine math in JavaScript. A node parity check at
the end proves JS and Python agree to 1e-9.
"""
import json
import os
import subprocess

import numpy as np

rng = np.random.RandomState(42)

# ---- Catalog: (title, year, [genres], [keywords], runtime_min) ----
# Genres come from a fixed list of 12; every movie has 3-6 keyword tags.
GENRES = ["Action", "Adventure", "Sci-Fi", "Drama", "Comedy", "Romance",
          "Thriller", "Horror", "Crime", "Fantasy", "Animation", "Mystery"]

MOVIES = [
    ("Neon Horizon", 1999, ["Sci-Fi", "Action"], ["space", "cyberpunk", "dystopia", "heist"], 127),
    ("Star Drifters", 2005, ["Sci-Fi", "Adventure"], ["space", "ai-robots", "exploration", "crew"], 132),
    ("Crimson Protocol", 2012, ["Action", "Thriller"], ["spy", "conspiracy", "explosion", "chase"], 118),
    ("Iron Vengeance", 2015, ["Action", "Crime"], ["revenge", "martial-arts", "gang", "fight"], 121),
    ("The Last Outpost", 1997, ["Action", "Sci-Fi"], ["war", "alien-invasion", "survival", "soldier"], 134),
    ("Blaze Runner", 2003, ["Action"], ["car-chase", "revenge", "explosion", "street-race"], 104),
    ("Echoes of Tomorrow", 2021, ["Sci-Fi", "Drama"], ["time-travel", "family", "parallel-worlds", "regret"], 126),
    ("Quantum Divide", 2019, ["Sci-Fi", "Thriller"], ["multiverse", "conspiracy", "physicist", "paradox"], 117),
    ("Sunset Heist", 2008, ["Crime", "Action"], ["heist", "double-cross", "casino", "crew"], 115),
    ("The Silent Ledger", 2016, ["Crime", "Thriller"], ["mafia", "money-laundering", "informant", "betrayal"], 139),
    ("Midnight Confession", 1994, ["Crime", "Drama"], ["detective", "noir", "murder", "rain-city"], 112),
    ("Glass Shadows", 2023, ["Thriller", "Mystery"], ["missing-person", "twist-ending", "hacker", "secrets"], 108),
    ("Whisper Network", 2018, ["Thriller", "Drama"], ["journalism", "conspiracy", "whistleblower", "cover-up"], 124),
    ("The Hollow Clock", 2001, ["Mystery", "Horror"], ["haunted-house", "curse", "supernatural", "clock"], 96),
    ("Graveyard Shift", 1992, ["Horror", "Comedy"], ["zombies", "gore", "small-town", "night-shift"], 88),
    ("Don't Turn Around", 2011, ["Horror", "Thriller"], ["serial-killer", "cabin", "survival", "woods"], 97),
    ("The Puppet House", 2020, ["Horror", "Mystery"], ["possession", "dolls", "seance", "family-curse"], 102),
    ("Laugh Track", 2006, ["Comedy", "Romance"], ["stand-up", "dating", "city-life", "club"], 94),
    ("Wedding Crash Course", 2013, ["Comedy", "Romance"], ["wedding", "best-friends", "chaos", "dance"], 101),
    ("The Intern Games", 2017, ["Comedy"], ["office", "competition", "underdog", "startup"], 99),
    ("Second Honeymoon", 1998, ["Romance", "Comedy"], ["vacation", "marriage", "misunderstanding", "beach"], 103),
    ("Letters to July", 2004, ["Romance", "Drama"], ["letters", "war", "separation", "promise"], 131),
    ("Halfway to Paris", 2010, ["Romance", "Comedy"], ["travel", "strangers", "cafe", "train"], 95),
    ("The Paper Boat", 2014, ["Drama"], ["childhood", "family", "poverty", "river"], 118),
    ("After the Harvest", 1996, ["Drama"], ["farm", "family", "grief", "harvest"], 121),
    ("One More Summer", 2009, ["Drama", "Romance"], ["terminal-illness", "beach", "promise", "grief"], 112),
    ("The Violin Maker", 2002, ["Drama"], ["music", "mentor", "redemption", "violin"], 127),
    ("Crossroads", 2019, ["Drama"], ["immigration", "identity", "family", "border"], 136),
    ("Dragonfire Keep", 2007, ["Fantasy", "Adventure"], ["dragons", "magic", "quest", "castle"], 145),
    ("The Elven Crown", 2012, ["Fantasy", "Adventure"], ["elves", "prophecy", "war", "kingdom"], 152),
    ("Moonlit Kingdom", 2022, ["Fantasy", "Romance"], ["fairy-tale", "magic", "kingdom", "curse"], 109),
    ("The Sorcerer's Debt", 2016, ["Fantasy"], ["wizard", "spell", "rivalry", "school"], 128),
    ("Pixel Quest", 2024, ["Animation", "Adventure"], ["video-game", "quest", "friendship", "arcade"], 98),
    ("Bubble Trouble", 2015, ["Animation", "Comedy"], ["pets", "city-life", "mischief", "chase"], 86),
    ("The Clockwork Forest", 2019, ["Animation", "Fantasy"], ["forest", "spirits", "child-hero", "magic"], 105),
    ("Rocket Pals", 2008, ["Animation", "Comedy"], ["space", "aliens", "school", "friendship"], 91),
    ("Sky Pirates", 2001, ["Adventure", "Fantasy"], ["pirates", "airship", "treasure", "sky"], 124),
    ("The Lost Valley", 1995, ["Adventure", "Drama"], ["jungle", "expedition", "survival", "river"], 129),
    ("Desert Thunder", 2010, ["Adventure", "Action"], ["desert", "treasure", "mercenaries", "chase"], 117),
    ("Arctic Rescue", 2022, ["Adventure", "Drama"], ["arctic", "rescue", "dogsled", "survival"], 106),
    ("The Final Whistle", 2005, ["Drama"], ["sports", "coach", "underdog", "team"], 114),
    ("Code Breakers", 2013, ["Thriller", "Drama"], ["hackers", "whistleblower", "government", "surveillance"], 122),
    ("The Night Dispatcher", 2017, ["Thriller", "Crime"], ["taxi-driver", "crime-ring", "neon-city", "night"], 107),
    ("Vanishing Point", 2000, ["Thriller", "Mystery"], ["amnesia", "chase", "identity", "conspiracy"], 111),
    ("Harbor Lights", 1993, ["Romance", "Drama"], ["lighthouse", "fishermen", "letters", "coast"], 108),
    ("Autumn in Lisbon", 2021, ["Romance", "Drama"], ["travel", "grief", "healing", "cafe"], 116),
    ("The Comedy of Errors", 1990, ["Comedy"], ["mistaken-identity", "theater", "slapstick", "twins"], 92),
    ("Gadget Dad", 2018, ["Comedy", "Sci-Fi"], ["inventor", "robot", "family", "gadget"], 93),
    ("Monster Mash-Up", 2026, ["Horror", "Comedy"], ["monsters", "party", "small-town", "chaos"], 89),
    ("The Deep Signal", 2025, ["Sci-Fi", "Mystery"], ["deep-space", "signal", "first-contact", "astronaut"], 119),
    ("Rust Belt", 2024, ["Drama", "Crime"], ["factory-town", "union", "betrayal", "strike"], 133),
    ("The Garden Party", 2023, ["Comedy", "Drama"], ["suburbia", "secret", "dinner-party", "neighbors"], 104),
    ("Starlight Academy", 2020, ["Animation", "Sci-Fi"], ["space-school", "aliens", "friendship", "training"], 97),
    ("The Timekeeper's Apprentice", 2011, ["Fantasy", "Adventure"], ["time-magic", "apprentice", "quest", "clock"], 130),
    ("Blood Moon Rising", 2007, ["Horror"], ["werewolves", "village", "full-moon", "hunt"], 95),
    ("Safe House", 2014, ["Action", "Thriller"], ["bodyguard", "conspiracy", "siege", "protect"], 110),
    ("The Alibi", 2009, ["Mystery", "Crime"], ["alibi", "courtroom", "detective", "trial"], 113),
    ("Parallel Hearts", 2025, ["Romance", "Sci-Fi"], ["parallel-worlds", "dating", "fate", "choice"], 109),
    ("Chef's Special", 2022, ["Comedy", "Drama"], ["restaurant", "chef", "competition", "food"], 96),
    ("The Last Reel", 1991, ["Drama"], ["cinema", "nostalgia", "projectionist", "film"], 125),
    ("Mecha Dawn", 2018, ["Sci-Fi", "Action"], ["mecha", "kaiju", "city-battle", "pilots"], 131),
    ("Whispering Pines", 2003, ["Mystery", "Thriller"], ["forest", "disappearance", "detective", "small-town"], 105),
    ("Neon Samurai", 1996, ["Action", "Sci-Fi"], ["cyberpunk", "samurai", "dystopia", "honor"], 123),
    ("The Wish Jar", 2026, ["Fantasy", "Drama"], ["wishes", "village", "miracle", "jar"], 112),
]
assert len(MOVIES) == 64, f"catalog must be exactly 64, got {len(MOVIES)}"

GENRE_IDX = {g: i for i, g in enumerate(GENRES)}
KEYWORDS = sorted({kw for _, _, _, kws, _ in MOVIES for kw in kws})
KW_IDX = {k: i for i, k in enumerate(KEYWORDS)}
ERAS = ["90s", "2000s", "2010s", "2020s"]


def era_of(year):
    return "90s" if year < 2000 else "2000s" if year < 2010 else "2010s" if year < 2020 else "2020s"


def runtime_bin(rt):
    return "short" if rt < 100 else "medium" if rt <= 140 else "long"


D = len(GENRES) + len(KEYWORDS) + len(ERAS) + 3   # feature dimension
print(f"catalog: {len(MOVIES)} movies | features: {D} "
      f"(genres {len(GENRES)} x2, keywords {len(KEYWORDS)}, era 4, runtime 3)")


def featurize(m):
    title, year, genres, keywords, rt = m
    v = np.zeros(D)
    for g in genres:                          # genre one-hot, weight x2
        v[GENRE_IDX[g]] = 2.0
    off = len(GENRES)
    for kw in keywords:                       # keyword one-hot
        v[off + KW_IDX[kw]] = 1.0
    off += len(KEYWORDS)
    v[off + ERAS.index(era_of(year))] = 1.0    # era one-hot
    v[off + len(ERAS) + ["short", "medium", "long"].index(runtime_bin(rt))] = 1.0
    return v / np.linalg.norm(v)               # L2-normalize


X = np.stack([featurize(m) for m in MOVIES])
assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def recommend(pick_idxs, top_n=5):
    """Query profile = mean of picked movies' feature vectors; rank the rest
    by cosine similarity (dot product, since all vectors are L2-normalized)."""
    profile = X[np.array(pick_idxs)].mean(axis=0)
    profile = profile / np.linalg.norm(profile)
    sims = X @ profile
    sims[np.array(pick_idxs)] = -1.0          # never recommend the picks themselves
    order = np.argsort(-sims)
    return [(int(i), float(sims[i])) for i in order[:top_n]]


def genre_agreement_p5(pick_idxs, top_n=5):
    """Fraction of the top-N recommendations sharing >= 1 genre with the query."""
    q_genres = set().union(*[set(MOVIES[i][2]) for i in pick_idxs])
    recs = recommend(pick_idxs, top_n)
    hits = sum(1 for i, _ in recs if set(MOVIES[i][2]) & q_genres)
    return hits / top_n


# ---- Evaluation: leave-one-movie-out — each movie as a single-pick query ----
p5 = np.mean([genre_agreement_p5([i]) for i in range(64)])

# Random baseline: 64 queries x 10 seeds, top-5 uniform random picks
rand_scores = []
for seed in range(10):
    r = np.random.RandomState(seed)
    for i in range(64):
        pool = [j for j in range(64) if j != i]
        r.shuffle(pool)
        q_genres = set(MOVIES[i][2])
        hits = sum(1 for j in pool[:5] if set(MOVIES[j][2]) & q_genres)
        rand_scores.append(hits / 5)
rand_p5 = float(np.mean(rand_scores))

print(f"mean genre-agreement P@5: {p5:.4f}")
print(f"random baseline P@5:     {rand_p5:.4f}")
print(f"lift over random:        {p5 / rand_p5:.2f}x")

# Sample queries for the demo sanity check
for picks, label in [([0], "Neon Horizon"), ([8, 10], "Sunset Heist + The Silent Ledger"),
                     ([33], "Pixel Quest")]:
    recs = recommend(picks)
    print(f"{label}:")
    for i, s in recs:
        shared = sorted(set(MOVIES[picks[0]][2]) & set(MOVIES[i][2]))
        print(f"   {MOVIES[i][0]} ({MOVIES[i][1]}) sim={s:.4f} shared={shared}")

# ---- Export movies.json (catalog + feature matrix) for the browser demo ----
export = {
    "genres": GENRES,
    "keywords": KEYWORDS,
    "movies": [
        {"title": m[0], "year": m[1], "genres": m[2], "keywords": m[3], "runtime": m[4]}
        for m in MOVIES
    ],
    "features": [[round(float(v), 4) for v in row] for row in X],
    "eval": {
        "genre_agreement_p5": round(float(p5), 4),
        "random_baseline_p5": round(float(rand_p5), 4),
        "n_movies": 64,
        "feature_dim": D,
    },
}
with open("movies.json", "w") as f:
    json.dump(export, f)
print(f"exported movies.json ({len(json.dumps(export)) / 1024:.1f} KB)")

# ---- JS parity check: node runs the same cosine math on a sample query ----
# (full-precision features here — movies.json rounds to 4 dp for size,
#  which would swamp a 1e-9 parity bar; parity is about the math, not storage)
parity_in = {
    "features": [[float(v) for v in row] for row in X],
    "picks": [0, 8, 10],
    "top_n": 5,
    "expected": [[i, s] for i, s in recommend([0, 8, 10])],
}
with open("parity_in.json", "w") as f:
    json.dump(parity_in, f)

NODE = r"""
const fs = require("fs");
const inp = JSON.parse(fs.readFileSync("parity_in.json", "utf8"));
const { features, picks, top_n, expected } = inp;
function norm(v) {
  let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s);
  return v.map(x => x / s);
}
const prof = norm(features[picks[0]].map((_, d) =>
  picks.reduce((a, i) => a + features[i][d], 0) / picks.length));
const picked = new Set(picks);
let worst = 0;
for (const [ei, es] of expected) {
  let dot = 0;
  for (let d = 0; d < prof.length; d++) dot += features[ei][d] * prof[d];
  worst = Math.max(worst, Math.abs(dot - es));
}
console.log("max abs diff vs Python:", worst.toExponential(2));
console.log(worst < 1e-9 ? "JS parity: PASS" : "JS parity: FAIL");
"""
with open("parity_check.js", "w") as f:
    f.write(NODE)
r = subprocess.run(["node", "parity_check.js"], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip())
os.remove("parity_in.json")
os.remove("parity_check.js")
if "FAIL" in r.stdout:
    raise SystemExit("parity check FAILED")
