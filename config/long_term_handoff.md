# Återupptagningsstatus: löpande långtidsanalys

Senast uppdaterad: 2026-09-07

Detta dokument är den operativa överlämningen för arbetet i
`config/long_term.md`. Börja här vid nästa arbetspass och kontrollera därefter
git-status och testresultat på nytt.

## Repon och arbetsläge

- Applikation: `/home/otyg/projects/feedSummary`
- Basbibliotek: `/home/otyg/projects/feedSummaryBase`
- Aktiv branch i båda repona: `feature/long-term-threat-analysis`
- Ändringarna är inte committade. Arbetskatalogerna innehåller även många
  orelaterade ospårade filer; stagea eller radera därför inte brett.
- Applikationens `requirements.txt` anger fortfarande
  `feedsummary-core[mongodb,ml]>=2.4.0`. En produktionsinstallation får inte
  antas innehålla långtidskoden förrän Base har fått en ny version eller appen
  har låsts till en exakt commit/version.

Senast verifierade testbaslinje:

- FeedSummaryBase: 176 tester godkända.
- FeedSummary: 102 tester godkända.
- Ruff är godkänd för de nya Base-modulerna, de fristående långtids-CLI:n och
  långtids-workertesterna. För den äldre workerfilen är import-, namn- och
  syntaxkontrollerna godkända; filen har sedan tidigare andra lintanmärkningar.
- Map- och rapport-CLI har körts framgångsrikt med `--dry-run`.

## Implementerat

### Fas 0 – analyskontrakt och kalibreringsstöd

- Read-only-kalibreringsverktyg för klustring.
- Versionssatt initial rapportprompt, JSON-schema och fem scenariefixtures.
- Versionssatta Map- och segmentprompter.

Manuellt märkt golden-dataset, uppmätta precision/recall-värden och dokumenterad
granskning av verkliga modellsvar återstår.

### Fas 1 – lagring

- Additiva kontrakt för cursor, lease, körningar, kluster, medlemskap,
  karantän, snapshots och rapporter.
- TinyDB, SQLite och MongoDB stöds.
- SQLite använder transaktioner för flerdokumentskrivningar. Dokumentlagren har
  återspelningsbar journal där det krävs.
- Snapshots och rapporter är insert-only och idempotenta.

### Fas 2 – deterministisk klustring

- Inkrementell, omkörningssäker processor med stabil cursorordning.
- Profilkällor, `required_tags`, karantän/retry, klusterlivscykel och
  återöppningar.
- Likhets-/dimensionskontroller och stark indikatorvakt; motstridiga CVE-ID:n
  får inte slås ihop.
- CLI med `--dry-run`, `--limit`, `--profile`, `--since` och `--until`.
- Embedding-backfill kan avgränsas till profil, tidsintervall och
  `similarity`-embedding.

### Fas 3 – Map

- Evidensbunden `cluster-update-v1` med strikt schema- och ID-validering.
- Deterministiskt urval av pending artiklar och kontrollerad inputbudget.
- Högst ett formatreparationsförsök.
- Atomisk/replaybar lagring av snapshot och `summarized_revision`.
- CLI med preview, limit, force och explicit val av lokal modell.

### Fas 4 – Reduce och publicering

- Deterministiska 7-/30-/90-dagarsmått på oberoende händelsekluster.
- Jämförelseperioder, veckobuckets, källtäckning, koncentration,
  källmixförskjutning och konservativ `trend_eligible`.
- Direkt eller segmenterad Reduce väljs utifrån kontextbudget.
- Strikt rapportvalidering blockerar okänd evidens, obelagda förändringar och
  prognoser vid otillräckligt underlag.
- Deterministisk Markdown och rapportidentitet bunden till prompt, modell,
  metrics och snapshots.
- Idempotent originalrapport i `threat_landscape_reports`.
- Deterministisk UI-spegel i `summary_docs`. Konflikter skrivs inte över och ett
  speglingsfel skadar inte originalet. Vanlig summary-cleanup kan radera spegeln
  men inte originalet.

### Fas 5 – worker, påbörjad

- Jobbtyperna `long_term_clustering`, `long_term_map`, `long_term_report` och
  `long_term_pipeline` stöds.
- CLI och worker använder samma körfunktioner och samma store-instans.
- Övergripande `long_term_run` innehåller aktuell fas, konfigurationshash,
  kompakt resultat och sanerat fel.
- Blockerad klustring stoppar senare pipelinefaser och lagras som `blocked`.
- Worker-API:t tillåter endast vitlistade långtids-overrides: `profile`,
  `since`, `until`, `at`, `limit`, `map_limit`, `cluster` och `force`.
- Fyra manuella `frequency: triggered`-exempel finns i
  `config/schedule.yaml.dist`.

## Blockerare före produktionstest

### 1. LLM-output och lease-livslängd

Detta är nästa kodsteg och högsta prioritet.

- `map_max_output_tokens` och `reduce_max_output_tokens` reserverar plats i
  tokenbudgeten men skickas inte som faktisk outputgräns via nuvarande
  `LLMClient.chat()` till Ollama.
- Exempelkonfigurationen tillåter `timeout_s: 6000`, medan Map-leasen är 1800
  sekunder och rapportleasen 3600 sekunder. En annan worker kan därför ta över
  en profil medan det första LLM-anropet fortfarande pågår.
- Lägg till en faktisk outputgräns i LLM-kontraktet och båda Ollama-klienterna,
  och vidarebefordra den genom fallback-klienten.
- Lägg till lease-förnyelse/heartbeat i lagringskontraktet och samtliga tre
  backends, eller bevisa med striktare timeouter att en lease alltid överlever
  hela anropet inklusive retries.
- Testa timeout, förnyelse, förlorad lease och samtidig takeover.

### 2. Failure-injection och end-to-end

Följande tester saknas fortfarande:

- två samtidiga workerstarter för samma profil,
- databasfel före och efter cursorflytt,
- LLM-timeout och fallback i Map respektive Reduce,
- omstart efter `blocked` och `failed`,
- modell-/embedding-signaturbyte mellan körningar,
- ett fast, nätverksfritt end-to-end-flöde från fixture-artiklar till rapport,
- prestandatest på ett representativt 180-/400-dagarsunderlag.

### 3. Produktionsdata och klusterkalibrering

- Kör embedding-audit mot hela produktionsprofilen.
- Gör en begränsad och säker `similarity`-backfill för saknade embeddings.
- Skapa och granska golden-datasetet.
- Mät precision, recall och tvetydiga fall innan tröskeln `0.80` godkänns.
- Granska karantän, felaktiga sammanslagningar och splittrade händelser efter en
  liten skrivande klusterbatch.

### 4. Verkliga LLM-regressionsfall

- Kör Map-, segment- och rapportprompterna mot exakt de modeller som ska användas.
- Spara anonymiserade svar som fixtures.
- Verifiera JSON-följsamhet, truncering, evidensreferenser, svenska,
  motstridiga källor och tunt underlag.
- Justera prompt först efter dokumenterat fel och höj promptversionen vid en
  kontraktsändring.

### 5. Paketering och databasberedskap

- Committa endast avsedda filer i båda repona.
- Publicera/versionera Base och lås appen till den versionen eller en exakt
  commit.
- Ta backup av produktionsdatabasen.
- Verifiera additiv schemainitialisering, unika index och store-kontrakt mot den
  verkliga MongoDB-versionen, inte bara mongomock.
- Lägg långtidsjobben i den faktiska produktions-schedulefilen; de finns ännu
  bara i distributionsmallen.

### 6. Shadow-konfiguration och driftinsyn

- Det finns ingen separat `shadow_mode`. Använd initialt
  `reports.mirror_to_summary_docs: false` så att rapporter inte visas i
  befintligt UI.
- Håll worker-API:t avstängt eller bundet till loopback. Det har ingen egen
  autentisering och får inte exponeras externt.
- Lägg minsta nödvändiga status/alerting för `failed`, `blocked`, lease-konflikt,
  karantänstorlek, snapshot-lag, cursor och LLM-körtid.
- `long_term_analysis.retention_days` finns i konfigurationen men verkställs
  ännu inte av cleanup. Implementera retention eller övervaka tillväxten under
  ett kort shadow-test; lös detta innan funktionen lämnas påslagen långvarigt.

## Rekommenderad återupptagningsordning

1. Implementera faktisk LLM-outputgräns genom klient- och fallbackkedjan.
2. Implementera lease-renewal/heartbeat för klustring, Map och Reduce.
3. Lägg failure-injection för timeout, lease-takeover och databasfel.
4. Lägg ett nätverksfritt end-to-end-test för hela pipelinen.
5. Skapa golden-dataset och kör kalibrering.
6. Kör produktionsdatans embedding-audit och en liten kontrollerad backfill.
7. Spela in verkliga Map-/Reduce-svar och lägg regressionsfixtures.
8. Versionera Base, lås appens beroende och verifiera deployment mot en kopia av
   produktionsdatabasen.
9. Kör shadow i ordningen klustring, manuell granskning, Map, manuell
   granskning, rapport utan spegling och först därefter full pipeline.
10. Låt shadow-körningen gå minst en vecka och godkänn först när den saknar
    cursorluckor/dubbletter och de analytiska resultaten har granskats manuellt.

## Första konkreta nästa arbetsuppgift

Utöka `LLMClient.chat()` med en valfri faktisk outputgräns utan att bryta
befintliga anrop. Vidarebefordra den genom `FallbackLLMClient`, mappa den till
Ollamas outputparameter i lokal- och cloud-klienterna och låt Map/Reduce skicka
sina konfigurerade gränser. Lägg enhetstester som visar att:

1. gränsen når den aktiva providern,
2. fallback får samma gräns,
3. gamla anrop utan gräns fungerar oförändrat,
4. en trunkerad/ogiltig respons aldrig persisteras,
5. kontextbudget och faktisk outputgräns använder samma värde.

Efter detta är nästa uppgift lease-renewal.

## Kommandon vid återupptagning

```bash
cd /home/otyg/projects/feedSummaryBase
git switch feature/long-term-threat-analysis
.venv/bin/python -m unittest discover -s tests

cd /home/otyg/projects/feedSummary
git switch feature/long-term-threat-analysis
.venv/bin/python -m unittest discover -s tests
```

Kontrollera alltid `git status --short` i båda repona innan redigering. Kör inte
bulk-stage eller cleanup av ospårade filer; flera av dem föregår detta arbete.

## Go/no-go för första shadow-testet

Starta inte ett skrivande produktionstest förrän:

- Base-versionen är entydigt installerad,
- databasbackup och indexverifiering är klara,
- outputgräns och lease-livslängd är säkrade,
- en liten embedding-backfill och klustergranskning är godkända,
- riktiga LLM-svar passerar validering utan okänd evidens,
- grundläggande failure-injection är grön,
- spegling till `summary_docs` är avstängd,
- status och larm för blockerade/felande körningar finns.

Fas 6, med fullständigt API/UI och klickbar evidens, är inte ett krav för
shadow-test men krävs före bred användarpublicering.
