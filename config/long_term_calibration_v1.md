# Initial klusterkalibrering v1

Datum: 2026-09-08

Status: **preliminärt arbetsunderlag – inte godkänt för produktion**.

## Underlag

- Profil: `healthcare_europe` från `config.yaml.dist`.
- Datakälla: lokal MongoDB, läst utan skrivningar.
- Urval: 2 000 profilmatchande artiklar med 100 procent embeddingtäckning.
- Embedding: `qwen3-embedding:0.6b`, 1 024 dimensioner, instruktionen
  `Represent the specific real-world event described by this article.`
- Golden-utkast: `config/long_term_golden_pairs_v1.csv`.
- Portabel fixture: `config/long_term_golden_fixture_v1.json`; 61 anonymiserade
  artiklar, 15 pseudonymiserade källor och float32-vektorer utan rubriker,
  råa artikel-ID:n eller ursprungliga indikatorvärden.
- Fixture-SHA256:
  `342dc4c941ce6656221f3ff55c9e21460c7a85e61a08a737d6dd2c971d30171a`.
- Två isolerade, syntetiska trepartsscenarier ingår: likvärdiga kandidater som
  ska ge `needs_review`, samt en CVE-konflikt som ska lämna en entydig matchning.
- Fixturen innehåller också 67 anonymiserade bakgrundsartiklar. För varje
  golden-artikel valdes högst tre tidigare icke-golden-artiklar inom 60 dagar
  och med cosinuslikhet minst `0.75`; efter deduplicering återstod 67.
- Omfattning: 33 par och 61 unika artiklar; 13 `same`, 15 `different` och
  5 `ambiguous`.

En riktad innehållsgranskning korrigerade AI-paret som tidigare märkts `same`:
artiklarna avser Hugging Face-incidenten respektive separata angrepp mot flera
AI-infrastrukturplattformar och är därför `different`. Samma granskning lade
till ett verkligt CareCloud-trepartsfall från tre källor, representerat av alla
tre parvisa `same`-relationer. Hela utkastet kräver fortfarande oberoende
fulltextgranskning.

Paren är stratifierade i hög likhet, nära tröskeln och nära negativa exempel.
De är därför avsedda att hitta fel vid beslutskanten, inte att uppskatta den
verkliga prevalensen av dubbletthändelser. Bedömningen är gjord från rubrik och
källkontext och måste verifieras oberoende mot fulltext innan den blir golden.

## Reproducerbart kommando

```bash
.venv/bin/python calibrate_long_term_clustering.py \
  --config config.yaml \
  --profile-config config.yaml.dist \
  --profile healthcare_europe \
  --limit 2000 \
  --pair-labels config/long_term_golden_pairs_v1.csv
```

Kommandot är read-only. Det genererar även en ny granskningsmall utan etiketter
med `--write-review-template PATH`, `--review-pairs-per-band`,
`--review-threshold` och `--ambiguity-margin`. En befintlig granskningsfil
skrivs aldrig över.

För oberoende fulltextgranskning av det befintliga golden-utkastet kan ett
blindat JSON-paket skapas:

```bash
.venv/bin/python calibrate_long_term_clustering.py \
  --config config.yaml \
  --profile-config config.yaml.dist \
  --profile healthcare_europe \
  --limit 2000 \
  --pair-labels config/long_term_golden_pairs_v1.csv \
  --write-pair-review-packet /säker/sökväg/long_term_pair_review.json
```

Paketet innehåller pseudonymiserade artikelalias, rubrik, källa, URL, fulltext,
parlikhet och tomma granskningsfält, men varken golden-bedömningar, anteckningar,
råa artikel-ID:n eller embeddings. Det innehåller upphovsrättsligt källmaterial
och ska därför inte committas. `--pair-review-text-chars` kan användas för ett
explicit texttak; `0` betyder full text. Befintlig fil skrivs aldrig över.

Den portabla fixturen reproducerar både cosinusmåtten och onlineklustringen utan
databas eller nätverk:

```bash
.venv/bin/python calibrate_long_term_clustering.py \
  --portable-dataset config/long_term_golden_fixture_v1.json \
  --thresholds 0.80,0.82 \
  --evaluate-online \
  --candidate-window-days 60 \
  --ambiguity-margin 0.03
```

## Observerade startvärden

Tvetydiga par ingår inte i precision/recall. Kolumnen "tvetydiga som samma"
visar hur många av de fem tvetydiga paren som passerar den rena
cosinuströskeln.

| Tröskel | Precision | Recall | F1 | Korrekt andel | Tvetydiga som samma |
|---:|---:|---:|---:|---:|---:|
| 0,70 | 0,464 | 1,000 | 0,634 | 0,464 | 5/5 |
| 0,75 | 0,464 | 1,000 | 0,634 | 0,464 | 5/5 |
| 0,78 | 0,684 | 1,000 | 0,813 | 0,786 | 4/5 |
| 0,80 | 0,813 | 1,000 | 0,897 | 0,893 | 1/5 |
| 0,82 | 1,000 | 1,000 | 1,000 | 1,000 | 0/5 |
| 0,85 | 1,000 | 1,000 | 1,000 | 1,000 | 0/5 |
| 0,90 | 1,000 | 1,000 | 1,000 | 1,000 | 0/5 |

Den faktiska kronologiska onlineklustringen på enbart de 61 fixtureartiklarna
gav:

| Tröskel | Kluster | Matchningar | Precision | Recall | Tvetydiga i samma kluster | `needs_review` |
|---:|---:|---:|---:|---:|---:|---:|
| 0,80 | 40 | 21 | 0,929 | 1,000 | 2/5 | 0 |
| 0,82 | 43 | 18 | 1,000 | 1,000 | 1/5 | 0 |

När de 67 frysta bakgrundskonkurrenterna inkluderades bearbetades totalt 128
artiklar:

| Tröskel | Kluster | Matchningar | Precision | Recall | FP | FN | `needs_review`-kluster |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,80 | 71 | 57 | 0,929 | 1,000 | 1 | 0 | 3 |
| 0,82 | 74 | 54 | 1,000 | 1,000 | 0 | 0 | 2 |

Vid `0.80` hamnade två separata vårdintrång (`article-028` och `article-060`)
i samma kluster. Innehållsgranskningen visar en centroidbrygga: det korrekta
CareCloud-trepartsfallet drar först in en separat Nutex Health-incident och
därefter SPRHS-incidenten. Vid `0.82` bryts kedjan eftersom Nutex-matchningen
ligger under tröskeln. Det tidigare AI-felet försvann efter korrigerad etikett;
det var ett etiketteringsfel, inte en falsk negativ. Resultatet reproduceras
från den portabla fixturen.

De två portabla trepartsscenarierna gav fyra av fyra godkända kontroller:

| Scenario | 0,80 | 0,82 | Bästa/andra likhet |
|---|---|---|---:|
| Likvärdiga kandidater | `needs_review` | `needs_review` | 0,866/0,866 |
| Motstridig CVE | korrekt `match` | korrekt `match` | 0,866/ingen kompatibel tvåa |

## Preliminär slutsats

Den rena cosinuströskeln `0.80` uppfyller inte precisionsmålet och samma
centroidbrygga syns i onlineklustringen. `0.82` är därför den starkare
preliminära kandidaten: den gav precision och recall 1,000 på samtliga 28
entydiga relationer, både med och utan bakgrund. Ingen tröskel är ännu godkänd
och konfigurationen ändras inte på detta lilla och riktade underlag.

Före beslut krävs:

1. oberoende fulltextgranskning av samtliga 33 par, särskilt de fem tvetydiga,
2. fler svåra negativa exempel med samma CVE, aktör, organisation eller sektor,
3. fler positiva par under `0.82` för att mäta risken för splittrade händelser,
4. fler verkliga fulltextgranskade trepartsfall, särskilt fall som ska utlösa
   `needs_review`; CareCloud-fallet verifierar hittills ett entydigt positivt
   trepartskluster,
5. oberoende kontroll av den korrigerade AI-etiketten och det observerade
   bryggklustret, följt av en större shadow-körning mot profilartiklarna.
