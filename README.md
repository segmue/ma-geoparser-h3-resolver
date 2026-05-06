# geoparser-h3-resolver

A [geoparser](https://github.com/dguzh/geoparser) resolver plugin that uses H3 spatial context to disambiguate toponyms. Instead of simple admin-hierarchy descriptions, it generates spatially informed sentences for the sentence-transformer model.

**Default output:**
`Matterhorn (Alpiner Gipfel) in Zermatt, Wallis, Wallis`

**This plugin:**
`Alpiner Gipfel "Matterhorn" bei Zmuttgrat, Hoernligrat (Grat); Theodulstrasse (Strasse). in Zermatt (Gemeinde), Wallis (Kanton)`

## How it works

1. **Build pipeline** (`spatial-h3-build`): Reads geometries from geoparser's SpatiaLite DB, converts them to H3 cells via [h3-multi-resolution-index](link), and computes spatial association matrices (B1/NPMI) between OBJEKTART categories.
2. **Sentence generator**: For each candidate toponym, finds spatially overlapping features and builds a descriptive sentence using association-ranked categories (dynamic context) and fixed slots like Gemeinde/Kanton (static context).
3. **Resolver** (`SpatialSentenceResolver`): Plugs into geoparser as a drop-in `SentenceTransformerResolver` — overrides `_generate_description()` with the spatial sentence generator.

## Usage

```python
from geoparser import Geoparser, SpacyRecognizer
from geoparser_h3_resolver import SpatialSentenceResolver

resolver = SpatialSentenceResolver(gazetteer_name="swissnames3d")
gp = Geoparser(recognizer=SpacyRecognizer(), resolver=resolver)
docs = gp.parse("Das Matterhorn liegt in den Walliser Alpen.")
```

## Setup

```bash
poetry install
poetry run spatial-h3-build              # builds DuckDB + association matrices
```

## Dependencies

geoparser, h3-multi-resolution-index, appdirs, pandas, numpy, pyyaml
