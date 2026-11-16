"""Generate clearly FAKE tiny dataset pairs to exercise every compare.py branch."""
import os
import random

import pandas as pd

OUT = "/tmp/gpqa_compare/synth"
rng = random.Random(0)

fake = [
    ("What colour is the zorblat crystal of planet Fnord?", "Mauve-ish", "Teal", "Ochre", "Plaid"),
    ("How many legs does a fictional quibbleworm have?", "Seventeen", "Three", "Nine", "Zero"),
    ("Which invented element has symbol Xq in the Snarf table?", "Quixonium", "Blorbium", "Zentite", "Fluffon"),
    ("In made-up Glarb physics, $E = \\frac{m}{c}$ means what?", "Energy over mass", "Mass flux", "Nothing", "Glarb drift"),
    ("What is the capital of the imaginary nation Wumpland?", "Wumptown", "Bleep City", "Zork Harbor", "Nowhere"),
    ("Which fake enzyme digests moonbeam pudding?", "Lunase-7", "Pudzyme", "Beamolase", "Gloopin"),
]


def official(rows):
    return pd.DataFrame([{"Record ID": f"rec{i}", "Question": q, "Correct Answer": c,
                          "Incorrect Answer 1": a, "Incorrect Answer 2": b,
                          "Incorrect Answer 3": d, "High-level domain": "Fake"}
                         for i, (q, c, a, b, d) in enumerate(rows)])


def mirror_row(q, c, a, b, d, shuffle=True, wrong=False, loose=False):
    opts = [c, a, b, d]
    if shuffle:
        rng.shuffle(opts)
    letter = "ABCD"[opts.index(c)]
    if wrong:
        letter = "ABCD"[(opts.index(c) + 1) % 4]
    if loose:
        q = q.upper().replace("$", "").replace("\\", "")
    s = q + "\n\n" + "\n".join(f"{L}) {o}" for L, o in zip("ABCD", opts))
    return {"question": s, "answer": letter}


def write(name, off, mir):
    d = os.path.join(OUT, name)
    os.makedirs(d, mode=0o700, exist_ok=True)
    off.to_csv(os.path.join(d, "official.csv"), index=False)
    pd.DataFrame(mir).to_parquet(os.path.join(d, "mirror.parquet"), index=False)


# 1) identical: same order, unshuffled options
write("identical", official(fake), [mirror_row(*r, shuffle=False) for r in fake])
# 2) reordered rows + shuffled options, all answers right
perm = [3, 0, 5, 1, 4, 2]
write("reordered", official(fake), [mirror_row(*fake[k]) for k in perm])
# 3) messy: row1 altered stem, row2 missing, row0 duplicated, row4 wrong letter,
#    row3 needs loose normalisation (case + LaTeX delimiters changed)
mir = []
mir.append(mirror_row(*fake[5]))
mir.append(mirror_row(*fake[0]))
alt = list(fake[1]); alt[0] = "How many arms does a fictional quibbleworm have?"
mir.append(mirror_row(*alt))
mir.append(mirror_row(*fake[3], loose=True))
mir.append(mirror_row(*fake[4], wrong=True))
mir.append(dict(mir[1]))  # exact duplicate of row0
write("messy", official(fake), mir)
print("synthetic pairs written")
