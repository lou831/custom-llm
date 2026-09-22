"""Generate the experiment-2 teaching files in corpus/ for two extension categories.

Grammar: subject/verb agreement (is/are/am, was/were) and verb tense
(walks / walking / walked / walk) with many different subjects.
Opposites: the frame "the opposite of X is Y" with pairs that are NOT tested,
plus ordinary contrast sentences for every adjective.

Separation rules (stricter than the notebook's exact-prefix check):
- no eval prompt appears anywhere (checked with the repository's own matcher);
- the tested pairs hot/cold, empty/full and noisy/quiet never share a sentence
  with the word "opposite", so the frame must transfer from other pairs;
- no extension test's last content word is ever directly followed by that test's
  answer (so no "bird is", "dogs are" or "she walked"); "bird" and "dogs" appear
  only in neutral sentences so the tests stay scorable;
- the text is written from scratch; nothing is copied from evals/ or results.

Run from the repository root:  python extension_corpus/make_extension_corpus.py
"""
import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_evals import load_suite, matching_cases, word_tokens  # noqa: E402

rng = random.Random(7)

# ---------------------------------------------------------------- grammar
singular = ["one cat", "one horse", "one child", "one farmer", "a cat", "a dog",
            "the cat", "the horse", "the girl", "the boy", "my sister", "my brother",
            "the baby", "our neighbor", "the old man", "he", "she", "tom", "lily"]
plural = ["two cats", "three horses", "many birds", "the cats", "the birds",
          "the girls", "the boys", "my friends", "our neighbors", "the children",
          "the farmers", "they", "we", "you", "tom and lily"]
states = ["happy", "tired", "hungry", "sleepy", "busy", "calm", "small", "ready", "outside", "at home"]
# verb: (base, third person, -ing, past, complement)
verbs = [("walk", "walks", "walking", "walked", "to the park"),
         ("cook", "cooks", "cooking", "cooked", "in the kitchen"),
         ("play", "plays", "playing", "played", "in the garden"),
         ("jump", "jumps", "jumping", "jumped", "over the fence"),
         ("clean", "cleans", "cleaning", "cleaned", "the room"),
         ("paint", "paints", "painting", "painted", "the wall"),
         ("visit", "visits", "visiting", "visited", "the farm"),
         ("climb", "climbs", "climbing", "climbed", "the hill"),
         ("wash", "washes", "washing", "washed", "the cup"),
         ("call", "calls", "calling", "called", "a friend"),
         ("talk", "talks", "talking", "talked", "with a friend"),
         ("travel", "travels", "traveling", "traveled", "to the city")]
third = [s for s in singular]              # singular subjects take walks / is / was
nonthird = plural + ["i"]                  # plural and "i" take walk (i takes am / was)
past_subjects = singular + plural + ["i"]

grammar = set()
for s in singular:
    for st in states:
        grammar.add(f"{s} is {st} .")
        grammar.add(f"{s} was {st} last night .")
for p in plural:
    for st in states:
        grammar.add(f"{p} are {st} .")
        grammar.add(f"{p} were {st} last night .")
for st in states:
    grammar.add(f"i am {st} .")
    grammar.add(f"i was {st} last night .")
for base, s3, ing, past, comp in verbs:
    for s in third:
        grammar.add(f"every day {s} {s3} {comp} .")
        grammar.add(f"right now {s} is {ing} {comp} .")
    for p in plural:
        grammar.add(f"every day {p} {base} {comp} .")
        grammar.add(f"right now {p} are {ing} {comp} .")
    grammar.add(f"every day i {base} {comp} .")
    grammar.add(f"right now i am {ing} {comp} .")
    for s in past_subjects:
        if s != "she":  # "yesterday she" is an eval prompt; never train on it
            grammar.add(f"yesterday {s} {past} {comp} .")
        grammar.add(f"{s} {past} {comp} yesterday .")
        grammar.add(f"tomorrow {s} will {base} {comp} .")
        grammar.add(f"last week {s} {past} {comp} .")

# "bird" and "dogs" appear only where no verb of agreement follows them.
for who in ["i", "we", "the children", "my sister", "tom"]:
    for where in ["in the garden", "near the river", "at the farm", "in the park"]:
        grammar.add(f"{who} saw a bird {where} .")
        grammar.add(f"{who} fed two dogs {where} .")

# --------------------------------------------------------------- opposites
# Pairs taught with the explicit frame. None of these are the tested pairs.
frame_pairs = [("big", "small"), ("fast", "slow"), ("early", "late"), ("heavy", "light"),
               ("soft", "hard"), ("wet", "dry"), ("tall", "short"), ("old", "young"),
               ("happy", "sad"), ("clean", "dirty"), ("thick", "thin"), ("strong", "weak"),
               ("high", "low"), ("rich", "poor"), ("warm", "cool"), ("round", "square"),
               ("bright", "dim"), ("smooth", "rough"), ("deep", "shallow"), ("sweet", "sour")]
# Tested pairs: contrast sentences only, never next to "opposite".
contrast_only = [("hot", "cold"), ("empty", "full"), ("noisy", "quiet"), ("loud", "soft")]

things = ["soup", "tea", "box", "bag", "room", "road", "stone", "bread", "river", "rope",
          "cup", "jar", "street", "hall", "bottle", "blanket", "car", "tree", "house", "basket"]
contrast_frames = [
    "the {t1} was {a} , but the {t2} was {b} .",
    "this {t1} is {a} and that {t2} is {b} .",
    "one {t1} felt {a} while the other {t1} felt {b} .",
    "in the morning the {t1} was {a} ; by night it was {b} .",
    "she wanted the {b} {t1} , not the {a} {t1} .",
    "the {a} {t1} and the {b} {t2} are very different .",
]

opposites = set()
for a, b in frame_pairs:
    opposites.add(f"the opposite of {a} is {b} .")
    opposites.add(f"the opposite of {b} is {a} .")
    opposites.add(f"{a} and {b} are opposites .")
    opposites.add(f"if something is not {a} , it may be {b} .")
for a, b in frame_pairs + contrast_only:
    for x, y in [(a, b), (b, a)]:
        for frame in contrast_frames:
            for t1, t2 in rng.sample(list(itertools.permutations(things, 2)), 6):
                opposites.add(frame.format(a=x, b=y, t1=t1, t2=t2))

# -------------------------------------------------------------- separation
suite = load_suite(ROOT / "evals" / "language_evals.json")
tested_words = {"hot", "cold", "empty", "full", "noisy", "quiet"}
function_words = {"is", "the", "a", "an", "of", "to", "into", "was", "from", "on", "at"}
banned_pairs = {(word_tokens(c["prompt"])[-1], c["answer"]) for c in suite["cases"]
                if c["group"] == "extend_corpus" and word_tokens(c["prompt"])[-1] not in function_words}

def has_banned_pair(sentence):
    tokens = word_tokens(sentence)
    return any(pair in banned_pairs for pair in zip(tokens, tokens[1:]))

for name, sentences in [("grammar", grammar), ("opposites", opposites)]:
    dropped = {s for s in sentences if has_banned_pair(s)}
    sentences -= dropped
    print(f"{name}: dropped {len(dropped)} sentences with a test word + answer pair")
for name, sentences in [("grammar", grammar), ("opposites", opposites)]:
    for sentence in sentences:
        assert not matching_cases(sentence, suite), (name, sentence)
        tokens = set(word_tokens(sentence))
        assert not ("opposite" in tokens or "opposites" in tokens) or not tokens & tested_words, sentence

out = ROOT / "corpus"
out.mkdir(exist_ok=True)
for name, sentences in [("grammar_practice.txt", grammar), ("opposites_practice.txt", opposites)]:
    lines = sorted(sentences)
    rng.shuffle(lines)
    (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {len(lines)} sentences")
