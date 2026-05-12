# Source Texts

Drop **60 English texts** here: 4 genres × 15 pieces. Each file is Markdown with YAML front-matter.

## Layout

```
source_texts/
├── academic/   15 files: arXiv abstracts / survey intros (public)
├── narrative/  15 files: public-domain short-story excerpts
├── essay/      15 files: public-domain essay excerpts (Montaigne, Emerson, ...)
└── poetry/     15 files: public-domain verse (Shakespeare sonnets, Dickinson, ...)
```

## Recommended acquisition workflow

We support a semi-automatic path aligned with the current experiment plan:

- `academic`: `gfissore/arxiv-abstracts-2021`
- `narrative`: `sanps/GutenbergFiction`
- `poetry`: `merve/poetry`
- `essay`: manual curation (Wikisource/public-domain essays)

Run:

```bash
python -m scripts.build_source_corpus --target-per-genre 15
```

This writes `academic_*`, `narrative_*`, and `poetry_*` automatically and creates
`essay/MANUAL_COLLECTION.md` with manual collection instructions.

## File format

Every file must begin with YAML front-matter and then the text:

```markdown
---
id: academic_07                 # unique; must match filename stem
genre: academic                 # one of: academic | narrative | essay | poetry
source: "arXiv:2310.12345"      # human-readable citation
license: public                 # public / CC-BY / your own
length: 218                     # word count, informational
note: "selected for explicit causal structure"
---
<150-300 words of source text, UTF-8, no headings>
```

## Constraints

- **Language**: English only.
- **Length**: 150–300 words per text. The framework enforces `length_ratio_min/max` at generation.
- **Content**: each text must have a clear propositional or imagistic payload (avoid pure dialogue, lists, or running summaries).
- **License**: prefer public-domain or your own original writing. Do NOT include copyrighted material without permission.

## Example file

See `academic/academic_01.md`, `narrative/narrative_01.md`, `essay/essay_01.md`, `poetry/poetry_01.md` for one example per genre. They are sufficient for the smoke test.

## Validation

After placing your files, run:

```bash
python -m src.run_experiment validate_corpus --expected-per-genre 15
```

This checks count (==60), length bounds, unique IDs, and required front-matter fields.
