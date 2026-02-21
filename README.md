# FeedSummary

FeedSummary är en liten pipeline + webbapp för att:
1) hämta artiklar via RSS,
2) extrahera brödtext,
3) batch-summera med en LLM,
4) skapa en metasammanfattning,
5) spara resultatet som `summary_docs` och visa i ett enkelt web-UI.

Färdiga moduler för att använda **Ollama Cloud** eller **Ollama Local**, med fallback-policy, som LLM finns men det är möjligt att lägga till moduler för andra LLMer.

Moduler för att lagra data i en flatfile JSON-databas (TinyDB) eller SQLite finns färdigt, men moduler för andra lagringsformat kan läggas till och användas.

---

## Snabbstart

### 1) Skapa config
Kopiera exempelkonfigen:

```bash
cp config.yaml.dist config.yaml
```

Justera minst:
- `llm.api_key` (om du kör `ollama_cloud`)
- paths för store/checkpoints om du vill

### 2) Installera beroenden
Repo:t innehåller inte alltid en låst requirements-fil, men koden använder typiskt:

- flask
- pyyaml
- markdown
- tinydb (om store=TinyDB)
- aiohttp
- feedparser
- trafilatura
- aiolimiter
- tenacity

Exempel:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### 3) Kör webbappen
Starta servern:

```bash
python webapp.py
```

Öppna sedan:
- http://127.0.0.1:5000

Du kan peka på en annan config med env-var:
```bash
FEEDSUMMARY_CONFIG=/path/till/config.yaml python webapp.py
```

---

## Hur UI:t fungerar

### Startsidan (`/`)
Startsidan visar:
- en sidomeny med tidigare sammanfattningar (från `summary_docs`)
- den valda sammanfattningen renderad som HTML (Markdown → HTML)
- en statusrad som visar körningsstatus (SSE)

#### Refresh-dialogen
Knappen **Refresh** öppnar en modal där du väljer:

1) **Tidsperspektiv (lookback)**
- t.ex. `24h`, `3d`, `1w` osv.
- styr vilka RSS-items som hämtas och vilka artiklar som väljs till sammanfattningen.

2) **Prompt-paket**
- väljer ett promptpaket ur `config/prompts.yaml`
- valet gäller *per körning* (UI skickar override)

3) **Ämnen (topics)** (om dina feeds har `topics`)
- markerar matchande källor automatiskt
- om du inte manuellt markerar källor kan körningen göras baserat på ämnen

4) **Källor (sources)**
- checkboxlista över feeds
- “Markera alla / Avmarkera alla”

När du trycker **Kör refresh** startas en pipeline i bakgrunden. UI lyssnar på status via SSE och uppdaterar statusrad.

### Artikelsidan (`/articles`)
Visar en lista över artiklar i store och har filter (infällbar panel):
- datumintervall (`from`/`to`)
- ämnen (topics)
- källor (sources)

Om du väljer ämnen markeras matchande källor i UI.
På servern gäller:
- om `sources` är valda → filtrera på exakt de källorna
- annars om `topics` är valda → härled tillåtna källor från config och filtrera på dem

---

## Pipeline i korthet

1) **Ingest**
- Hämtar RSS via `feedparser`
- Filtrerar per lookback (t.ex. 1 dygn)
- Extraherar brödtext med `trafilatura`
- Sparar artiklar i store (`articles`)

2) **Urval**
- Väljer artiklar för sammanfattning baserat på:
  - lookback
  - valda källor (eller härledda via ämnen)
  - (store kan ha `list_articles_by_filter` för effektivare urval)

3) **Summering**
- Artiklar batchas (max antal / max chars)
- För varje batch: LLM skapar batch-sammanfattning
- Meta-steget bygger en övergripande sammanfattning från batch-sammanfattningarna
- Resultatet sparas som `summary_doc`

4) **Ämnesindelade sammanfattningar (valfritt/om aktiverat i koden)**
Om du har topic-baserad pipeline:
- artiklar grupperas per “primärt ämne” (första taggen på källan)
- en sammanfattning körs per ämnesområde
- slutresultatet blir ett dokument med en sektion per ämne
- `summary_doc` innehåller både `summary` (hela dokumentet) och `sections[]` (per ämne)

---

## Konfiguration

### `config.yaml`
Baseras ofta på `config.yaml.dist`.

#### `store`
Väljer lagring. Exempel TinyDB:

```yaml
store:
  provider: tinydb
  path: ~/.local/share/FeedSummary/news_docs.json
```

Om du har SQLite-stöd:
```yaml
store:
  provider: sqlite
  path: ~/.local/share/FeedSummary/news_docs.sqlite
```

#### `checkpointing`
Checkpoint/resume för långkörningar:

```yaml
checkpointing:
  enabled: true
  dir: ~/.local/share/FeedSummary/checkpoints
```

#### `feeds`
Pekar ut feeds-filen:

```yaml
feeds:
  path: "config/feeds.yaml"
```

#### `ingest`
Ingest-beteende:

```yaml
ingest:
  lookback: 1d
  max_items_per_feed: 100
  article_timeout_s: 20
```

- `lookback`: hur långt bak RSS-items tas med.
- `max_items_per_feed`: safety cap per feed.
- `article_timeout_s`: timeout när en artikel hämtas.

#### `batching`
Batch- och meta-budgets:

```yaml
batching:
  max_articles_per_batch: 15
  max_chars_per_batch: 14500
  article_clip_chars: 3500
  meta_batch_clip_chars: 1500
  meta_sources_clip_chars: 100
  retry_user_clip_chars: 9000
```

- `max_articles_per_batch`: max artiklar i en batch.
- `max_chars_per_batch`: max textmängd per batch (tecken).
- `article_clip_chars`: klipper varje artikeltext.
- `meta_*`: styr hur mycket som får plats i meta-steget.

#### `llm` och `llm_fallback`
Primary LLM och fallback:

```yaml
llm:
  provider: ollama_cloud
  host: https://ollama.com
  model: gemma3:27b-cloud
  api_key: CHANGE-ME
  context_window_tokens: 24576
  max_output_tokens: 500
  prompt_safety_margin: 1600
  token_chars_per_token: 2.4
  prompt_too_long_max_attempts: 6
  prompt_too_long_structural_threshold_tokens: 1200
  quota:
    preflight: true
    min_interval_seconds: 2

llm_fallback:
  provider: ollama_local
  model: gemma3:1b
  base_url: http://localhost:11434
  max_rps: 1
  timeout_s: 6000
  sock_read_timeout_s: 360
  max_retries: 3
  retry_backoff_s: 2.0
  context_window_tokens: 24576
  max_output_tokens: 500
  prompt_safety_margin: 1600
  token_chars_per_token: 2.4
  prompt_too_long_max_attempts: 6
  prompt_too_long_structural_threshold_tokens: 1200
```

Viktiga begrepp:
- `context_window_tokens`: total plats för input+output.
- `max_output_tokens`: hur långt svaret får bli.
- `prompt_safety_margin`: buffert för att undvika att slå i context-taket.

---

### `config/feeds.yaml`
Listar RSS-källor. Minsta fält:
- `name`: källnamn (används som `source`)
- `url`: RSS/Atom URL

Exempel:
```yaml
- name: SVT
  url: https://www.svt.se/rss.xml
```

Valfria fält:

#### `topics`
Ämnestaggar för UI-snabbval och ämnesindelad summering:
```yaml
- name: CERT-SE
  url: https://www.cert.se/feed.rss
  topics: ["Cyber", "Sårbarheter", "Sverige"]
```

> Om ämnesindelning används blir *första* topic ofta “primär”.

#### `category_include` / `category_exclude`
Filter per feed baserat på RSS-entry tags/kategorier:
```yaml
- name: TV4
  url: https://www.tv4.se/rss
  category_include: ["Inrikes", "Utrikes"]
```

---

### `config/prompts.yaml`
Innehåller prompt-paket (”packages”). Varje package är en nyckel i YAML och innehåller fyra fält:

- `batch_system`
- `batch_user_template`
- `meta_system`
- `meta_user_template`

Exempelstruktur:

```yaml
MyPackage:
  batch_system: |
    ...
  batch_user_template: |
    ... {articles_corpus} ...
  meta_system: |
    ...
  meta_user_template: |
    ... {batch_summaries} ...
```

Vanliga placeholders:
- `{articles_corpus}`: injiceras i batch-steget (artiklar + metadata)
- `{batch_summaries}`: injiceras i meta-steget

I `config.yaml` väljer du default/selected:
```yaml
prompts:
  path: "config/prompts.yaml"
  default_package: "SecurityAnalyst"
  selected: ""   # tom => default; webapp kan override per körning
```

---

## Tips & felsökning

- Om inga artiklar kommer med:
  - öka `ingest.lookback`
  - kontrollera att dina feeds svarar och att artikelsidorna går att hämta
- Om LLM klagar på för lång prompt:
  - minska `batching.max_chars_per_batch`
  - minska `batching.article_clip_chars`
  - öka `llm.context_window_tokens` (om modellen stödjer)
  - eller minska `llm.max_output_tokens` och/eller öka `prompt_safety_margin`
- Om UI visar “Status-anslutning bröts”:
  - refresh-sidan reloadas normalt när jobbet blir `done`
  - annars kontrollera serverloggar

---

## Licens
BSD 3-Clause (se headers i källfilerna).

## Special shout out
- [C. Strömblad](https://cstromblad.com/) för inspiration till detta lilla projekt