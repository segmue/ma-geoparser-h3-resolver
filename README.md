# geoparser-h3-resolver

Resolver plugin for [geoparser](https://github.com/dguzh/geoparser). It describes each candidate by nearby gazetteer objects, chosen via category associations estimated on an H3 grid.

- Default: `Säntis (Alpiner Gipfel) in Schwende-Rüte, Hinterland, Appenzell Ausserrhoden`
- Plugin: `Säntis, Alpiner Gipfel, bei Alpstein, Massiv, Flis und Säntis-Nordwand, Gebiet, Obertoggenburg und Toggenburg, Landschaftsname, in Schwende-Rüte, Gemeinde, Hundwil, Gemeinde, Hinterland, Bezirk, Appenzell Ausserrhoden, Kanton`

## Setup

```bash
poetry install
poetry run spatial-h3-build               # H3 index + B1 matrix (overlap)
poetry run spatial-h3-assoc --measure d1  # optional: D1 matrix (adjacency)
```

## Usage

```python
from geoparser import Geoparser
from geoparser.modules import SpacyRecognizer
from geoparser_h3_resolver import SpatialSentenceResolver

gp = Geoparser(recognizer=SpacyRecognizer(model_name="de_core_news_sm"),
               resolver=SpatialSentenceResolver())
docs = gp.parse("Der Säntis liegt im Alpstein.")
```

## Configuration

`geoparser_h3_resolver/configs/swissnames3d.yaml`: association measure (`association.measure: b1 | d1`) and sentence generator parameters.
