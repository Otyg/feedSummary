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

## Disclaimer

Det var längesen jag kodade något större och även om jag förespråkar TDD så lever jag inte som jag lär. Koden är
väldigt mycket på formen "just works" och innehåller både dupliceringar, döda funktioner och allmänt håriga konstruktioner.

You've been warned...

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

### Read-only-variant

Den separata read-only-appen återanvänder samma templates och läs-API:er. Alla
registrerade tagg- och kategoriändringar svarar med HTTP 401. Worker-status,
återupptagning och schematrigger finns inte i denna variant.

```bash
python -m web_viewer.webapp_viewer_readonly --config config.yaml --port 5001
```

För WSGI kan appen laddas som
`web_viewer.webapp_viewer_readonly:app`. `FEEDSUMMARY_CONFIG` stöds på samma
sätt som i den vanliga viewern.

### RSS

Det finns två RSS 2.0-feeds som fungerar i både den vanliga viewern och
read-only-varianten:

- `/rss/summaries.xml` för sammanfattningar
- `/rss/articles.xml` för artiklar

Antalet poster kan begränsas med exempelvis `?limit=50` (högst 500).

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
- väljer ett promptpaket ur katalogen `config/prompts/`
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
  extraction:
    path: "config/extraction.yaml"
```

- `lookback`: hur långt bak RSS-items tas med.
- `max_items_per_feed`: safety cap per feed.
- `article_timeout_s`: timeout när en artikel hämtas.
- `extraction.path`: valfri YAML-fil med domänspecifika extraktionsregler.

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

#### `tagging`

Taggning av färdiga sammanfattningar:

```yaml
tagging:
  summary_max_tags: 20
  summary_include_cve_tags: false
```

- `summary_max_tags`: max antal vanliga taggar; CVE-taggar räknas inte in när de är aktiverade.
- `summary_include_cve_tags`: sätt till `false` för att behålla CVE-ID:n i
  sammanfattningstexten utan att skapa separata CVE-taggar.

#### `llm`
LLM-konfiguration anges som en lista där första posten är primary och efterföljande poster används som fallback i den ordning `feedsummary_core` förväntar sig:

```yaml
llm:
  - provider: ollama_cloud
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

  - provider: ollama_local
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

### `config/extraction.yaml`

Kan begränsa Trafilatura till ett särskilt HTML-element för domäner där den
generella brödtextextraktionen tar med oönskat innehåll:

```yaml
domains:
  theregister.com:
    content_xpath: "//main/article/section[1]"
```

Domännamn matchas exakt, bortsett från att `www.` normaliseras bort. Om XPath-
selektorn är ogiltig, inte ger någon träff eller fragmentet inte ger någon text,
loggas en varning och Trafilatura körs på hela sidan. Följande valfria
Trafilatura-inställningar kan också anges per domän: `include_comments`,
`include_tables`, `include_links`, `include_images`, `favor_precision`,
`favor_recall`, `deduplicate`, `target_language` och `prune_xpath`.

---

### `config/prompts/`
Innehåller prompt-paket (”packages”). Varje paket ligger i en egen YAML-fil, där filnamnet utan ändelsen är paketnamnet, och innehåller fyra fält:

- `batch_system`
- `batch_user_template`
- `meta_system`
- `meta_user_template`

Exempel, `config/prompts/MyPackage.yaml`:

```yaml
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
  path: "config/prompts"
  default_package: "daily_cyber_multisource_se_eu_world_mod"
  selected: ""   # tom => default; webapp kan override per körning
```

---

## Granska befintliga taggkopplingar

`audit_article_tags.py` använder projektets konfiguration, store och LLM-klient för att
kontrollera samtliga befintliga taggkopplingar automatiskt. Helt okopplade taggposter
listas också i rapporten. Standardläget är read-only och kräver inga taggnamn:

```bash
.venv/bin/python audit_article_tags.py --output tag-audit.json
```

Granska rapporten och kör därefter vid behov med `--remove-invalid` för att ta bort de
taggkopplingar som LLM-bedömningen markerar som irrelevanta. Efter kopplingssaneringen
tas även taggposter som helt saknar artikelkoppling bort:

```bash
.venv/bin/python audit_article_tags.py \
  --input-report tag-audit.json \
  --output tag-audit-cleanup.json \
  --remove-invalid
```

Med `--input-report` återanvänds den tidigare rapportens bedömningar och inga nya
LLM-anrop görs. Rapportens artikel-ID, tagg-ID och taggnamn verifieras mot den aktuella
databasen före varje borttagning.

### Granska synonymer i en taggkategori

`audit_tag_synonyms.py` går interaktivt igenom varje synonym i en vald kategori.
En synonym kan behållas, hoppas över eller omvandlas till en egen barntagg i samma
kategori.

```bash
.venv/bin/python audit_tag_synonyms.py \
  --config config.yaml \
  --category LOCATION \
  --output tag-synonym-audit.json
```

När en barntagg skapas körs det befintliga förslagsflödet på artiklar som har
föräldrataggen. Endast taggar ur den valda kategorin visas för modellen. Om barnet
föreslås läggs det till innan föräldrataggen tas bort. Kategorin matchas
skiftlägesokänsligt. Förhandsgranska besluten utan databasändringar eller LLM-anrop med
`--dry-run`. Det äldre kommandot `audit_location_synonyms.py` finns kvar och använder
`LOCATION` när `--category` utelämnas.

## Träna om produktionsmodellen helt

Efter en ändring under `tagging.ml` i konfigurationen kan produktionsmodellen byggas
om från samtliga lagrade artikel-embeddings och taggkopplingar:

```bash
.venv/bin/python retrain_tagging_ml.py --config config.yaml
```

Scriptet använder MongoDB- och modellinställningarna från konfigurationen och hoppar
över den vanliga kontrollen av corpus-fingeravtrycket. Modellfilen ersätts atomiskt
först när den nya träningen har lyckats. Om träningsunderlaget inte uppfyller
`min_training_articles` eller `min_label_support` returneras ett fel och den befintliga
modellen behålls. `--config` kan utelämnas; då används `FEEDSUMMARY_CONFIG` eller
`config.yaml`.

## Benchmarka lättvikts-ML för taggning

`benchmark_tagging_ml.py` är fristående från produktionsflödet. Verktyget läser
befintliga artiklar och taggkopplingar från den MongoDB som anges i `config.yaml`,
jämför etablerade scikit-learn-algoritmer på samma kronologiska datasplit och skriver
både JSON och Markdown. Det aktiverar eller sparar ingen produktionsmodell.

```bash
.venv/bin/python benchmark_tagging_ml.py \
  --config config.yaml \
  --output-dir benchmark_results/tagging_ml
```

Som standard jämförs Logistic Regression, SGD, två Naive Bayes-varianter, K-Nearest
Neighbour, Random Forest och linjär Support Vector Machine. Rapporten innehåller en
körning med alla `DEFAULT_CATEGORIES` tillsammans, en körning för varje kategori och
parvisa körningar med `DOMAIN_ENTITY` och var och en av de övriga kategorierna.
JSON- och Markdownrapporterna rankar den bästa algoritmen och representationen för
varje sådan kombination efter micro-F1. En annan baskategori kan anges med
`--combination-base-category`. `--categories LOCATION,THREAT` lägger till manuella
kategorier efter standardkategorierna; det ersätter dem inte.

Varje kategoriomfång körs med TF-IDF, hashade ord-/teckenfeatures, befintliga
artikel-embeddings och en hybrid av TF-IDF och embeddings. Före tidsdelningen begränsas
varje kategoriomfång till artiklar med samma kompatibla, redan sparade embeddingmodell
och dimension. Därmed använder samtliga representationer exakt samma artiklar och
etiketter. Rapporten redovisar modell, dimension, täckning och bortfall. Verktyget
skapar inga nya embeddings. En viss modell kan krävas med `--embedding-model`, och
representationer kan begränsas med exempelvis
`--representations tfidf,embedding,hybrid`.

Ett mindre prov kan köras med exempelvis
`--max-articles 200 --algorithms logistic_regression,linear_svm`.

## Granska produktionsmodellens DOMAIN_ENTITY-förslag

Den tränade produktionsmodellen kan köras skrivskyddat mot samtliga artiklar och
jämföras med databasens faktiska `DOMAIN_ENTITY`-taggar:

```bash
.venv/bin/python benchmark_ml_domain_entity.py \
  --config config.yaml \
  --output-dir benchmark_results/ml-domain-entity-audit
```

Verktyget laddar den befintliga modellartefakten utan att träna om den, genererar
inga embeddings och ändrar aldrig databasen. `report.json` innehåller samtliga
artiklar och maskinläsbara skillnader. `report.md` visar täckning, precision/recall/F1,
jämförelse per tagg samt artiklar med möjliga taggtillägg eller modellmissar.
Artiklar utan kompatibel embedding redovisas separat. Använd `--limit 20` för en
snabb provkörning.

Måtten är en in-sample-audit eftersom modellen har tränats från samma databas.
Föreslagna tillägg ska därför granskas manuellt och faktiska taggar som modellen
missar ska inte tas bort automatiskt.

Auditbedömningen tar hänsyn till taggens konfigurerade synonymer och till överordnade
geografiska begrepp, exempelvis `Europe` för en artikel vars centrala plats är ett
europeiskt land. Rapporten anger `match_type` och `matched_term`. Vid applicering av en
äldre rapport utan synonympolicyn hoppas taggar med synonymer över för säkerhets skull.

Använd `--limit 10` för en mindre provkörning. `--config` kan ange en annan konfiguration;
annars används `FEEDSUMMARY_CONFIG` eller `./config.yaml`. Taggnamn kan fortfarande anges
som positionella argument om bara ett urval ska granskas, exempelvis `ray comfast`.

---

## Tips & felsökning

- Om inga artiklar kommer med:
  - öka `ingest.lookback`
  - kontrollera att dina feeds svarar och att artikelsidorna går att hämta
- Om LLM klagar på för lång prompt:
  - minska `batching.max_chars_per_batch`
  - minska `batching.article_clip_chars`
  - öka `context_window_tokens` för den aktiva posten i `llm` (om modellen stödjer)
  - eller minska `max_output_tokens` och/eller öka `prompt_safety_margin`
- Om UI visar “Status-anslutning bröts”:
  - refresh-sidan reloadas normalt när jobbet blir `done`
  - annars kontrollera serverloggar

---

## Licens
BSD 3-Clause (se headers i källfilerna).

## Special shout out
- [C. Strömblad](https://cstromblad.com/) för inspiration till detta lilla projekt
