# Återupptagningsstatus: löpande långtidsanalys

Senast uppdaterad: 2026-09-08

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

- FeedSummaryBase: 198 tester godkända.
- FeedSummary: 105 tester godkända.
- Ruff är godkänd för de nya Base-modulerna, de fristående långtids-CLI:n och
  långtids-workertesterna. För den äldre workerfilen är import-, namn- och
  syntaxkontrollerna godkända; filen har sedan tidigare andra lintanmärkningar.
- Map- och rapport-CLI har körts framgångsrikt med `--dry-run`.

Korrigering 2026-09-08: promptloadern söker rekursivt under `config/prompts`.
Långtids-fixtures ligger därför i den privata katalogen
`config/prompts/long_term/_fixtures`, som loadern ignorerar. Lägg inte fixtures
i en vanlig underkatalog med samma filnamn som ett riktigt promptpaket; det
blockerar även ordinarie summeringsjobb med ett duplicate-package-fel.

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
- `LLMClient.chat()` har en bakåtkompatibel, valfri faktisk outputgräns.
  `FallbackLLMClient` vidarebefordrar samma gräns vid retry/fallback och både
  lokal och cloud-Ollama mappar den till `num_predict`. Map, segment-Reduce,
  final-Reduce och deras formatreparationer skickar samma `max_output_tokens`
  som reserveras i respektive kontextbudget. Anrop utan gräns skickar ingen ny
  providerparameter.
- `renew_long_term_lease()` är atomisk och ägarkontrollerad i TinyDB, SQLite
  och MongoDB och vägrar återuppliva en utgången eller övertagen lease. Map och
  Reduce har bakgrunds-heartbeat under LLM-anrop och verifierar ägarskapet igen
  omedelbart före snapshot-/rapportskrivning. Klustringen förnyar under batchen
  och verifierar före cursor-commit. Databasfel eller leaseförlust stoppar
  persistens, och release kan inte radera en ny ägares lease.
- Failure-injection visar att exakt en av två samtidiga klusterstarter får
  profilleasen, att fel både före och efter cursor-commit kan köras om utan
  dubblett eller cursorlucka, och att uttömd timeout-fallback i Map/Reduce inte
  skapar snapshot, rapport eller UI-spegel. Ett fast nätverksfritt end-to-end-
  test kör tre fixture-artiklar genom klustring, Map, mått, Reduce, validering
  och publicering i SQLite.

## Blockerare före produktionstest

### 1. Failure-injection och end-to-end

Följande tester saknas fortfarande:

- omstart efter `blocked` och `failed`,
- modell-/embedding-signaturbyte mellan körningar,
- prestandatest på ett representativt 180-/400-dagarsunderlag.

Lease-renewal, expiry, ägarbyte, databasfel i heartbeat, skydd före persistens
och leaseförlust under ett långsamt Map-anrop är enhetstestade. Samtidig start,
cursorfel, uttömd LLM-fallback och ett nätverksfritt helflöde är också testade.
Ovanstående återstarts-, signatur- och prestandafall återstår.

### 2. Produktionsdata och klusterkalibrering

- Kör embedding-audit mot hela produktionsprofilen.
- Gör en begränsad och säker `similarity`-backfill för saknade embeddings.
- Skapa och granska golden-datasetet.
- Mät precision, recall och tvetydiga fall innan tröskeln `0.80` godkänns.
- Granska karantän, felaktiga sammanslagningar och splittrade händelser efter en
  liten skrivande klusterbatch.

### 3. Verkliga LLM-regressionsfall

- Kör Map-, segment- och rapportprompterna mot exakt de modeller som ska användas.
- Spara anonymiserade svar som fixtures.
- Verifiera JSON-följsamhet, truncering, evidensreferenser, svenska,
  motstridiga källor och tunt underlag.
- Justera prompt först efter dokumenterat fel och höj promptversionen vid en
  kontraktsändring.

### 4. Paketering och databasberedskap

- Committa endast avsedda filer i båda repona.
- Publicera/versionera Base och lås appen till den versionen eller en exakt
  commit.
- Ta backup av produktionsdatabasen.
- Verifiera additiv schemainitialisering, unika index och store-kontrakt mot den
  verkliga MongoDB-versionen, inte bara mongomock.
- Lägg långtidsjobben i den faktiska produktions-schedulefilen; de finns ännu
  bara i distributionsmallen.

### 5. Shadow-konfiguration och driftinsyn

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

1. Slutför failure-injection för omstart och embedding-signaturbyte.
2. Lägg ett representativt 180-/400-dagars prestandatest.
3. Skapa golden-dataset och kör kalibrering.
4. Kör produktionsdatans embedding-audit och en liten kontrollerad backfill.
5. Spela in verkliga Map-/Reduce-svar och lägg regressionsfixtures.
6. Versionera Base, lås appens beroende och verifiera deployment mot en kopia av
   produktionsdatabasen.
7. Kör shadow i ordningen klustring, manuell granskning, Map, manuell
   granskning, rapport utan spegling och först därefter full pipeline.
8. Låt shadow-körningen gå minst en vecka och godkänn först när den saknar
    cursorluckor/dubbletter och de analytiska resultaten har granskats manuellt.

## Första konkreta nästa arbetsuppgift

Testa omstart efter en `blocked` klusterkörning där artikeln senare har
korrigerats/backfillats samt efter ett `failed` Map-/Reduce-anrop. Verifiera att
statushistorik, cursor, klustermedlemskap och insert-only-dokument förblir
idempotenta. Lägg därefter ett signaturbyte där embeddingmodell, dimension eller
instruktion ändras mellan batcher; äldre och nya signaturer får inte blandas i
samma kluster och körningen ska ge ett tydligt granskningsbart utfall.

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
