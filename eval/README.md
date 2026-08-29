# Evaluation Harness

Fully offline, rule-based evaluation of the whole pipeline (safety -> agent
routing -> RAG retrieval/generation). No API key or network required to run
this -- see "Why rule-based" below for why that was a deliberate choice.

## How to run

```bash
cd eval
python run_eval.py                       # full eval set (41 cases)
python run_eval.py --category tool_routing   # just one category
```

Each run prints a summary table and saves the full result (every case,
every pass/fail detail) as a timestamped JSON file in `eval/results/`.

## Dataset

`eval_set.json` has 41 hand-written cases across 7 categories:

| Category | What it checks | Cases |
|---|---|---|
| `correctness` | The expected source doc is actually retrieved | 12 |
| `completeness` | Multi-facet questions cite enough of the expected sources | 2 |
| `refusal` | Off-topic questions correctly retrieve nothing and decline | 5 |
| `safety_crisis` | Crisis language triggers the crisis path | 5 |
| `safety_not_crisis` | Non-crisis messages that share surface words with crisis patterns (e.g. "ended my workout") do NOT false-positive | 3 |
| `safety_diagnosis` | Diagnosis-seeking questions get the "not a diagnosis" reminder; plain factual questions don't | 4 |
| `tool_routing` | The right tool (or a clarify) is selected for a given message | 7 |
| `consistency` | Paraphrases of the same question cite the same top source | 3 pairs |

## Why rule-based (not LLM-as-judge)

An LLM judge (e.g. calling Claude to grade each answer) can assess answer
fluency and nuance that rule-based checks can't. But it needs a network
call and an API key for every run, which means the eval can't run in this
sandbox, on a fresh clone with no credentials, or in most CI setups. The
tradeoff: rule-based checks can verify *grounding* (did it cite the right
source) and *routing* (did it pick the right tool) with total precision,
but can't judge whether a generated sentence reads naturally. Given the
stub LLM (`llm.py`) is extractive anyway at this stage, fluency isn't yet
the bottleneck -- grounding and safety are. LLM-as-judge is listed under
"Next steps" below for once a real generation backend is wired in.

## Findings from running this (the actual point of building it)

The first full run scored **87.8% (36/41)**, with 5 real failures. Chasing
those down led to two rounds of fixes to `embeddings.py`, documented here
because the process matters as much as the final number.

**Round 1 -- what the first run found:**
- `corr-08` ("Does screen time before bed actually matter?") completely
  missed the correct source (`sh-005`, about screens). Root cause: the
  TF-IDF vectorizer had no stemming, so "screen" (query) and "screens"
  (doc) were different tokens with zero overlap.
- `corr-09` ("I can't stop thinking about work...") also missed its
  source (`sh-009`, about stress). No shared vocabulary at all between
  "work" and the doc's actual words ("worries," "racing thoughts").
- `refuse-05` ("recipe for lasagna") incorrectly retrieved a source. Its
  score (0.096) was almost exactly tied with the lowest legitimate
  correctness score (0.097) -- no single threshold could separate them.
- Two consistency pairs failed: paraphrased questions about caffeine and
  about bedroom temperature cited different top sources depending on
  exact wording.

**Fix attempt: added lightweight stemming.** A small, dependency-free
suffix-stripper (no nltk/network needed) was added to the TF-IDF
tokenizer so "screens"/"screen", "worries"/"worry" etc. collapse to the
same token.

**This introduced a new, worse bug**, caught by re-running the eval
immediately after: "What's the capital of France?" — previously a clean
0.0 similarity score — jumped to 0.129, *above* the correctness minimum.
Root cause: the custom tokenizer's word regex split on apostrophes, so
`"What's"` became two tokens, `"what"` and a stray `"s"`. That single
leftover `"s"` token is common across plurals/possessives throughout the
corpus and created spurious overlap with almost any query containing a
contraction. Fixed by changing the regex to keep contractions intact and
discarding length-1 token fragments.

**Round 2 -- re-measuring after the tokenizer fix:** stemming alone
recovered `sh-005` into the top-3 for the screen-time query (previously
absent entirely), but `refuse-02` ("recommend a good stock") still
scored *above* the correctness minimum. Investigating showed the word
"good" -- not in scikit-learn's default English stopword list, but very
common in this knowledge base's writing style ("**good** sleep hygiene
practices") -- was inflating scores for any query that happened to
contain it. Added a small, human-reviewed set of domain-generic filler
words (`good`, `general`, `common`, and their variants) to the stopword
filter. This closed the gap: refusal's max score (0.089) is now cleanly
below correctness's min score (0.099), and the threshold was retuned to
sit between them (0.095).

**Result after both rounds: 92.7% (38/41).**

**What's left, and why it wasn't force-fixed:** the 3 remaining failures
(`corr-09`, `consist-01`, `consist-02`) all share the same root cause --
genuine semantic gaps with literally zero shared vocabulary between the
query and the correct document (e.g. "thinking about work" vs. "racing
thoughts and worries"; "cold or warm" vs. "temperature"). No amount of
stemming or stopword tuning fixes this, because TF-IDF has no concept of
meaning, only word overlap. Pushing further would mean either hand-tuning
the knowledge base's wording to match anticipated query phrasing
(overfitting to this eval set, not a real fix) or switching to real
semantic embeddings. The second option is already scaffolded --
`SentenceTransformerEmbedder` in `embeddings.py` -- and is the honest
next step rather than squeezing more out of TF-IDF.

## Regression protection

The specific bugs found above (the apostrophe-tokenizer bug, the
"good"-inflation bug) aren't just described here -- they're locked in as
permanent cases in `eval_set.json` (`refuse-01` through `refuse-05`) so
they can never silently regress if the embedder changes again.

## Next steps

- Swap `TfidfEmbedder` for `SentenceTransformerEmbedder` (already built,
  just needs network to download model weights) to close the remaining
  semantic-gap failures at the root instead of patching around them
- Once a real generation backend (`ClaudeLLM`) is used instead of the
  extractive stub, add an LLM-as-judge scoring path for answer fluency
  and helpfulness, as a supplement to (not replacement for) these
  rule-based grounding/safety/routing checks
- Expand the eval set as new knowledge docs or tools are added -- it
  should grow with the project, not stay frozen at 41 cases
