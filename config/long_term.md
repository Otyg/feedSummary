# Genomförandeplan: löpande långtidsanalys av hotlandskapet

> Operativ status, produktionsblockerare och exakt återupptagningsordning finns
> i [`long_term_handoff.md`](long_term_handoff.md).

## Implementationsläge 2026-09-10

- Grenen `feature/long-term-threat-analysis` finns i både applikations- och
  basbiblioteksrepot.
- Den initiala cloud-baselinen är färdig. Den sista Map-passen sparade samtliga
  19 återstående kluster utan fel och Reduce skapade shadow-rapporten
  `threat_report_72ea6f4fdae356c49a0bb82bbd926dff` från 219 snapshots.
  Två revisionsposter som tillkom genom timklustringen efter rapportens frysta
  tidpunkt har också Map-behandlats; en tvingad preview visar nu tom Map-kö.
  Rapporten är sparad i långtidslagret men har, enligt
  `mirror_to_summary_docs: false`, ingen vanlig summary-spegel.
- Fas 0 är påbörjad med read-only-kalibreringsverktyg, versionssatt initial
  analysprompt, JSON-schema och fem prompt-fixtures. Ett första stratifierat
  golden-utkast med 33 par och explicita `same`/`different`/`ambiguous`-etiketter
  finns tillsammans med en portabel anonymiserad fixture och stöd för att skapa
  ett blindat fulltextpaket för en oberoende granskare. Ren cosinus vid
  `0.80` gav preliminärt precision 0,813, medan den faktiska onlineklustringen
  gav precision/recall 0,929/1,000 för `0.80` och 1,000/1,000 för `0.82`.
  Ingen körning av de produktionshärledda paren utlöste `needs_review`. Två
  isolerade
  trepartsscenarier verifierar dock både tvetydighetsgrenen och CVE-vakten vid
  båda trösklarna. En riktad innehållsgranskning korrigerade ett felmärkt
  AI-par och lade till ett verkligt CareCloud-trepartsfall. Med 67 frysta
  bakgrundskonkurrenter gav `0.80` en falsk sammanslagning och precision 0,929,
  medan `0.82` gav precision och recall 1,000. Trösklarna får inte godkännas
  före oberoende fulltextgranskning, fler låg-likhetspositiva exempel och verkliga
  trepartsfall. Dokumenterad promptgranskning återstår innan fasen är godkänd.
- Fas 1 har additiva lagringsmodeller, cursor/lease, körningsposter, kluster,
  medlemskap, snapshots och rapporter för TinyDB, SQLite och MongoDB. Den
  kombinerade kluster-/medlemskapsskrivningen är transaktionell i SQLite och
  återspelningsbar via journal i dokumentlagren; TinyDB:s avbrottsreparation är
  kontraktstestad.
- Fas 2 har en första deterministisk, omkörningssäker batchprocessor och ett CLI
  med `--dry-run`, `--limit`, `--profile`, `--since` och `--until`. Cursor flyttas
  endast efter bestående medlemskap eller karantänpost. Öppna karantänposter
  återprövas automatiskt och löses när artikeln senare kan klustras. Profilens
  `required_tags`, tidsstyrd livscykel och en första stark indikatorvakt är
  implementerade. Motstridiga CVE-ID:n är ett hårt stopp endast när båda
  sidorna är rena en-CVE-katalogposter; narrativa händelseartiklar kan matcha
  trots olika CVE-mängder. En read-only reconciliation-pass föreslår dessutom
  sammanslagningar av befintligt fragmenterade kluster utan att ännu flytta
  medlemskap eller invalidera snapshots.
- Embedding-backfill kan nu begränsas till långtidsprofil, tidsintervall och
  `similarity`-embedding samt köras read-only. En verklig dry-run mot tre lokala
  profilartiklar identifierade tre saknade embeddings utan LLM-anrop eller
  databasskrivning. Nästa operativa steg är en kontrollerad similarity-backfill
  följd av shadow-klustring och granskning av karantän-/klusterutfall.
- Fas 3 har `cluster-update-v2` (med `cluster-update-v1` bevarad), fem
  prompt-fixtures, en körbar Map-tjänst och
  strikt snapshotvalidering av schema, identitet, medlemsrevision, MITRE-format
  och samtliga evidensreferenser. Pending artiklar väljs deterministiskt i
  begränsade delar, promptens inputbudget kontrolleras före LLM-anrop och högst
  ett formatreparationsförsök görs. Snapshot och `summarized_revision` sparas
  atomiskt i SQLite och återspelningsbart via journal i dokumentlagren. Map
  använder 8 000 outputtokens i aktuell cloud-konfiguration. Innehållslösa
  uncertainty-markörer tas bort evidensneutralt medan strukturerade objekt
  bevaras förlustfritt som deterministisk JSON-text. Ett CLI stöder
  read-only-preview och explicit lokal modell. Inspelade verkliga
  LLM-regressionsfall återstår; Map-körningen är inkopplad i Fas 5:s workerflöde.
- Fas 4 är påbörjad med reproducerbara 7-/30-/90-dagarsmått som räknar
  oberoende händelsekluster separat från artiklar. Underlaget innehåller
  jämförelseperioder, veckobuckets, status och återöppningar, källtäckning,
  koncentration, källmixskifte samt konservativ `trend_eligible`. Ett read-only
  CLI kan frysa beräkningen vid en explicit tidpunkt. Reduce-kontraktet validerar
  nu identitet, period, konfigurerad evidensmängd och faktisk spridning över
  tidsbuckets samt blockerar förändringar och prognoser när inget fönster är
  trendbehörigt. Reduce tillämpar dessutom denna grind deterministiskt före
  validering, så att modellen inte kan kringgå den genom att ändå returnera
  `changes` eller `forecast`. Markdown renderas deterministiskt och rapport-ID:t binds till
  prompt, segmentprompt, modell och fryst input. En körbar Reduce-tjänst väljer
  automatiskt direkt eller segmenterad syntes utifrån kontextbudgeten, tillåter
  högst en formatreparation och sparar först efter full validering. Rapport-CLI:t
  har lease, `--dry-run`, fryst körtid och idempotent publicering till det
  separata långtidslagret. I `publish` kan original speglas deterministiskt och
  idempotent till `summary_docs`, vilket betyder en intern extra kopia i den
  vanliga FeedSummary-listan/RSS och inte publicering på internet; konflikt
  skrivs inte över och ett partiellt
  speglingsfel kan köras om mot det redan sparade originalet. Vanlig
  summary-retention får gallra UI-spegeln men inte långtidsoriginalet; den
  separata långtidsretentionen gallrar först när dess längre fönster löpt ut.
  Reduce skickar JSON-schema till segment-, reparations- och finalanrop, gör
  upp till tre lokala genereringsförsök för ett felande segment och loggar
  segmentprogress. En kompakt promptvy utelämnar icke-citerbara rå-ID-
  upprepningar men bevarar numeriska mått och tidsbucket-kopplingar. Ett
  deterministiskt slutsteg behåller varje citerbart kluster-ID och anpassar
  textdetaljen till inputbudgeten. Originalmåtten används oförändrade för
  validering och persistens. Inspelade verkliga LLM-regressionsfall återstår.
- Fas 5 är påbörjad med workerjobb för separat klustring, Map, rapport och en
  sammanhållen pipeline. Jobben återanvänder CLI-kodvägarna, kräver aktiverad
  feature-flagga och profil, samt sparar övergripande status med fas,
  konfigurationshash, kompakt resultat och sanerat fel i `long_term_runs`.
  Trigger-API:t tillåter endast en vitlistad mängd långtids-overrides. Den
  faktiska LLM-outputgränsen går nu genom fallbackkedjan till lokal och cloud
  Ollamas `num_predict`, och Map/Reduce använder samma gräns som sina
  tokenbudgetar. Ägarkontrollerad lease-renewal finns i samtliga backends;
  Map/Reduce använder bakgrunds-heartbeat och kontrollerar leasen före
  persistens, medan klustringen förnyar under batchen och före cursor-commit.
  Samtidig start, cursorfel före/efter commit och uttömd timeout-fallback är
  failure-injection-testade. Ett nätverksfritt helflöde täcker fixture-artiklar
  genom klustring, Map, mått, Reduce och publicerad rapport. Återstart efter
  `blocked`/`failed` bevarar historiken och kan fortsätta idempotent;
  embedding-signaturbyten blandas inte mellan kluster. Ett reproducerbart,
  nätverksfritt benchmark för 90-/180-dagarsunderlag mäter kandidaturval, mått,
  snapshoturval, Reduce-input, total körtid och minnestopp; standardlasten ligger
  under de satta lokala gränserna. En första verklig shadow-körning är
  genomförd; minst en veckas driftmått och manuell resultatgranskning återstår.
  `long_term_analysis.retention_days` verkställs nu i TinyDB, SQLite
  och MongoDB. Kvarvarande rapporter skyddar refererade snapshots och deras
  kluster; endast utgångna stängda kluster och deras medlemskap gallras.
  Workern loggar separata borttagningsräknare för shadow-verifiering.
  Ett explicit `mode: shadow` blockerar nu alltid spegling till `summary_docs`
  även om speglingsflaggan skulle vara felsatt. Den aktiva, git-ignorerade
  pilotkonfigurationen använder explicit `cloud` för både Map och Reduce,
  tröskel `0.82` och batch 25. `cloud` filtrerar bort alla `ollama_local`-poster
  ur dessa två LLM-kedjor; `configured_chain` finns kvar som ett separat val.
  Fyra manuella jobb kompletteras av klustring varje timme, Map var sjätte timme
  och shadow-rapport måndag 05:30. En skrivande pilotbatch har nått totalt 50
  medlemskap i 49 kluster utan karantän eller `needs_review`. Fem Map-snapshots
  skapades först via cloud-kedjan utan formatreparation. En större
  återupptagningsbar Map-körning sparade ytterligare 23 innan ett ogiltigt
  cloud-svar stoppades säkert; totalt finns 28 snapshots och 21 ännu ej
  berättigade singelkluster. Den första riktiga rapporten skapades därefter från
  28 snapshots i 6 segment och 7 cloud-anrop, utan formatreparation. Den sparades
  som `threat_report_a9d3f30b82325e5791013db138d16ff9` i
  `threat_landscape_reports`; shadow-läget gav `mirror_action: disabled`.
  Slutprompten uppskattades till 15 086 tokens inom budgeten 20 476.
- Fas 6 är enligt beslut 2026-09-08 påbörjad parallellt med kvarvarande
  produktionsvalidering. Första read-only-snittet ligger bakom den
  separata flaggan `long_term_analysis.ui.enabled`, så shadow-rapporter kan
  granskas utan att `long_term_analysis.enabled` behöver slå på nya
  pipelinekörningar. Ett versionerat API och serverrenderade vyer visar nu
  rapportlista, rapportdetalj, profiler, perioder, kvalitetsvarningar, tomläge
  och prognoshypoteser tydligt skilda från observerade fakta. Provenienskedjan
  rapport -> snapshot -> kluster -> artikel är nu klickbar. Klustervyn visar
  bevarad titel, källa, URL och publiceringstid även när originalartikeln har
  gallrats. Synlig länktext använder snapshot-/klustertitel, kompletterad med
  belagd CVE när sådan finns, samt artikelrubrik i stället för interna ID:n;
  ID:n finns kvar i URL och teknisk proveniens. Centroid och vektorsumma
  exponeras aldrig av API:t. `/long-term/clusters` visar nu en paginerad
  klusteröversikt med profil- och statusfilter, mänskliga rubriker, medlemsantal,
  senaste observation och väntande snapshotrevisioner. Motsvarande versionerade
  läs-API är `/api/v1/long-term/clusters`; listfrågan hämtar inte lagrade
  embedding-vektorer. Från klusterdetaljen kan användaren öppna ett
  profilavgränsat taggnätverk och styra djupet uppåt och nedåt. Uppåt används de
  existerande klustren vars taggmängder är strikta delmängder. Varje ny nivå
  väljer den största tillgängliga subset-storleken, så ett hopp kan ta bort flera
  taggar men taggantalet är gemensamt inom nivån. Nedåt används samma relations
  exakta invers, vilket gör att varje visad föräldrakant också kan följas tillbaka
  till barnet. Vyn har konfigurerbara gränser för djup, noder och inlästa kandidater
  och finns även i read-only-viewern. Ett separat artikelnätverk placerar
  basklustret i centrum och viktar grannkluster efter antalet unika gemensamma
  artikelmedlemskap. Alla visade klusterpar med överlappning förbinds och läggs
  ut med repulsion, kollisionsavstånd och viktad attraktion. Andelen gemensamma
  artiklar med basklustret låser radialavståndet monotont, medan grannklustrens
  inbördes kopplingar påverkar deras vinklar. En koncentrisk reservlayout bevarar
  samma ordning och lika avstånd för lika överlappning. Minsta procentuella
  överlappning och maximalt antal grannar kan styras i vyn. Basrelationer använder
  basklustrets artikelantal som nämnare; övriga kanter använder Jaccard-andelen.
  Ett första granskningsflöde finns via
  `/long-term/review/<cluster_id>` för äldre granskningskluster: artikeln jämförs
  med kandidaterna och den skrivbara viewern kan registrera `merge`,
  `keep_separate` eller `exclude`. Besluten är revisionskontrollerade och
  loggar granskare, tid och motivering; merge återanvänder det atomiska
  reconciliation-kontraktet. Nya tvetydiga tilldelningar sparas i flera kluster;
  äldre granskningsposter rekonstrueras och märks tydligt som sådana. Skrivning
  kräver den separata flaggan `long_term_analysis.ui.review_actions_enabled`;
  read-only-viewern blockerar alltid åtgärderna. Olösta och exkluderade
  granskningskluster ignoreras nu i nya trendmått. Nästa UI-steg är paginering för mycket stora klusters
  medlemslistor och slutlig granskning av
  åtkomst, navigation och deployment i read-only-viewern. Bred
  produktionsaktivering är fortsatt blockerad av oberoende golden-granskning,
  embedding-audit/backfill, verkliga Map/Reduce-regressioner, databas-/indexkontroll
  och minst en veckas godkänd shadow-körning.

## 1. Mål och avgränsning

Målet är att bygga en inkrementell, omkörningsbar och granskningsbar analys ovanpå
FeedSummarys befintliga artikelpipeline. Systemet ska:

- följa nya och förändrade cyberhändelser över tid,
- skilja flera artiklar om samma händelse från flera oberoende händelser,
- sammanställa belagda mönster över rullande tidsfönster,
- skapa en källbunden långtidsanalys och försiktig prognos,
- bevara tillräcklig proveniens för att varje slutsats ska kunna granskas,
- kunna återuppta en avbruten körning utan dubbla medlemskap eller rapporter.

Den första versionen är ett analytiskt beslutsunderlag, inte ett autonomt
varningssystem eller ett sannolikhetskalibrerat prediktionssystem. Den ska inte
tolka fler artiklar som bevis för högre hotaktivitet och ska inte skapa
MITRE ATT&CK-mappningar, attribueringar eller prognoser utan stöd i underlaget.

## 2. Nuläge och beslut som ersätter den ursprungliga planen

Planen är anpassad till kodbasen så som den ser ut 2026-09-07:

- `feedsummary_core.persistence.NewsStore` är lagringsgränssnittet. TinyDB,
  SQLite och MongoDB stöds; den aktiva installationen använder MongoDB.
- Artiklar har sträng-ID:n, Unix-tidsstämplar och text i `text`, inte det
  MongoDB-/ObjectId-schema som den tidigare planen antog.
- Det finns redan två separata artikelvektorer. Långtidsanalysen ska återanvända
  `similarity_embedding_*`, vars instruktion representerar den specifika
  verkliga händelsen. `tagging_embedding_*` används fortsatt för klassificering.
- Aktuella artiklar har 1024-dimensionella `qwen3-embedding:0.6b`-vektorer, men
  modellnamn, dimension och embedding-instruktion ska alltid behandlas som en
  versionssignatur och aldrig hårdkodas.
- Schemaläggaren stöder `hourly`, `quarterday`, `halfday`, `daily`, `weekly` och
  `triggered`; den stöder inte godtyckliga 15-minutersintervall. Nuvarande
  `continuous_tagging` hämtar och taggar varje timme.
- Ordinarie gallring tar bort artiklar efter 180 dagar. Långtidslagret måste
  därför ha en egen retention och bevara minsta nödvändiga evidensmetadata.
- `create_llm_client()` väljer den ordinarie fallback-kedjan, som kan börja i
  Ollama Cloud. Om Map-steget ska vara lokalt måste det väljas uttryckligen i
  konfigurationen; ett visst modellnamn ska inte ligga i koden.
- Nuvarande `LLMClient.chat()` tar bara temperatur som anropsoption. Stöd för
  strukturerat svar och en faktisk outputgräns behöver antingen läggas till i
  klientgränssnittet eller hanteras av en avgränsad långtidsanalysklient.

Följande arkitekturbeslut gäller:

1. Implementera mot `NewsStore`, inte direkt mot `MongoDBStore`. MongoDB kan
   optimera kandidatfrågor senare, men ska inte vara ett funktionskrav.
2. Dela upp analysen i **händelsekluster** och **trend-/landskapssyntes**.
   Exempelvis är en specifik ransomwareincident ett händelsekluster, medan
   "ransomware mot europeisk sjukvård" är ett mönster som måste beläggas av
   flera oberoende händelser. Dessa får inte vara samma objekt.
3. Spara medlemskap som separata, unika poster. En växande `article_ids`-lista i
   klusterdokumentet ger dålig idempotens, konkurrenshantering och historik.
4. Uppdatera en aktuell klustersammanfattning men bevara oföränderliga snapshots.
   Ny information får markera äldre uppgifter som ersatta eller bestridda, inte
   radera den historiska observationen.
5. Generera maskinläsbar, validerad JSON först och rendera Markdown från den.
   Fri Markdown ensam är inte tillräcklig för proveniens eller regressionstest.
6. Flytta artikelmarkören först efter en lyckad transaktion/batch. Samma
   artikelmängd ska kunna spelas om med samma resultat.

## 3. Målarkitektur

```text
RSS -> artikel + taggar + similarity_embedding
                         |
                         v
             inkrementell artikelväljare
               (profil + vattenmärke)
                         |
                         v
              händelseklustring
          (matcha / skapa / karantän)
                         |
                         v
          strukturerad klusteruppdatering
             + oföränderlig snapshot
                         |
                         v
       deterministiska tidsserier och deltan
                         |
                         v
       landskapssyntes + prognoshypoteser
                         |
                         v
       rapport, evidensgraf och kvalitetsdata
```

En profil med `aggregate_profiles` är ett orkestrerings- och rapportlager. Den
ska inte skapa en parallell uppsättning Map-sammanfattningar. Vid körning
uppdateras barnprofilerna idempotent, deras senaste snapshots projiceras till
samlingsprofilens rapportunderlag och identiska kluster dedupliceras utifrån
embedding-signatur, taggscope och artikelmedlemskap. Originalens snapshot- och
kluster-ID:n bevaras för spårbarhet.

Misslyckade barnprofiluppdateringar stoppar inte samlingsrapporten. Senast
tillgängliga underlag används och rapporten markeras som ofullständig med
`quality.completeness: partial`, `aggregate_profile_update_failed` och en
strukturerad fellista. Varningarna ingår också i Reduce-prompten så att modellen
inte behandlar ofullständig täckning som fullständig evidens.

En **analysprofil** avgränsar exempelvis geografi, sektorer, källor och
rapporthorisont. Samma artikel kan ingå i flera profiler, men högst en gång per
profil. Första profilen bör motsvara nuvarande fokus på cyberhot, Europa och
hälso- och sjukvård. Profil-ID ingår i alla unika nycklar så att fler profiler
kan läggas till utan migrering.

Kodens ansvar delas på samma sätt som i resten av projektet:

- `feedSummaryBase/src/feedsummary_core/long_term/`: modeller, validering,
  klustring, Map/Reduce-tjänster och Markdown-rendering,
- `feedSummaryBase/src/feedsummary_core/persistence/`: backend-neutralt kontrakt
  samt TinyDB-, SQLite- och MongoDB-implementationer,
- `feedSummary/web_viewer/feedsum_worker.py`: schemaläggning och jobbstatus,
- `feedSummary/config/`: avstängda standardvärden, profil och promptpaket,
- respektive repos `tests/`: kontrakts-, domän- och integrationstester.

## 4. Domänmodell och lagring

Alla tidpunkter lagras som UTC Unix-sekunder för att följa befintlig kod. Alla
ID:n är stabila strängar. Dokumenten har `schema_version` och genererade objekt
har även `algorithm_version` respektive `prompt_version`.

ID-strategin ska göra retries idempotenta: ett nytt kluster-ID härleds från
profil, första artikel-ID och algoritmversion; snapshot-ID från kluster,
medlemsrevision och promptversion; rapport-ID från profil, fryst periodslut och
rapportversion. Använd en namnbaserad UUID eller motsvarande stabil hash, inte
aktuell klocka som enda identitet.

### 4.1 Utöka `NewsStore`

Lägg domänspecifika metoder i protokollet och implementera kontraktet för
TinyDB, SQLite och MongoDB:

- `list_articles_for_long_term(profile_id, after_cursor, until_ts, limit)`
- `claim_long_term_run(...)`, `finish_long_term_run(...)`
- `get_long_term_cursor(profile_id)` och atomisk `advance_long_term_cursor(...)`
- `get_candidate_threat_clusters(profile_id, embedding_signature, ...)`
- `save_threat_cluster(...)` och `update_threat_cluster_if_revision(...)`
- `get/save_cluster_membership(...)`
- `list_pending_cluster_memberships(...)`
- `save/list_cluster_snapshots(...)`
- `save/list/get_threat_landscape_reports(...)`

Det publika protokollet ska vara backend-neutralt. Backend-specifika index och
bulkoperationer får ligga i respektive implementation.

Minsta index/unika nycklar är:

- kluster: `(profile_id, status, last_seen_ts)` samt embedding-signatur,
- medlemskap: unikt `(profile_id, article_id)` och index på
  `(cluster_id, assigned_at)`,
- snapshots: unikt `(cluster_id, membership_revision, prompt_version)` och
  index på `(profile_id, created_at)`,
- rapporter: unikt rapport-ID och index på `(profile_id, period_end_ts)`,
- cursor/lease: unikt `profile_id`,
- körningar: `(profile_id, run_type, started_at)` och `status`.

### 4.2 `threat_clusters`

Ett kluster representerar en sammanhållen incident, kampanj eller utveckling:

```yaml
id: string
profile_id: string
schema_version: 1
status: active | dormant | closed
name: string
first_seen_ts: int
last_seen_ts: int
created_at: int
updated_at: int
embedding_model: string
embedding_dimension: int
embedding_instruction: string
centroid: list[float]
vector_sum: list[float]
member_count: int
membership_revision: int
summarized_revision: int
last_summarized_at: int | null
reopened_count: int
last_reopened_at: int | null
summary_markdown: string
summary_structured: object
facets: object
quality_flags: list[string]
```

`vector_sum` och `member_count` krävs för en exakt inkrementell centroid. Kluster
med annan embedding-signatur än artikeln är aldrig kandidater. Byte av modell,
dimension eller instruktion kräver en explicit omembedding och omklustrering,
inte tyst blandning av vektorrum.

### 4.3 `threat_cluster_memberships`

```yaml
profile_id: string
article_id: string
cluster_id: string
article_ts: int
ingested_ts: int
assigned_at: int
similarity: float | null
decision: matched | new_cluster | manual
embedding_signature: string
embedding_source_hash: string
evidence:
  title: string
  url: string
  source: string
  published_ts: int
```

Unik nyckel är `(profile_id, article_id)`. Den lilla `evidence`-kopian gör
historiken begriplig efter ordinarie artikelgallring; artikeltexten dupliceras
inte. Om full reproducerbarhet längre än 180 dagar krävs ska
`cleanup.articles_days` höjas till minst rapportens maximala baslinje plus
säkerhetsmarginal, exempelvis 400 dagar för en ettårsanalys.

### 4.4 `threat_cluster_snapshots`

Varje lyckad Map-uppdatering skapar en oföränderlig snapshot med:

- kluster-ID och medlemsrevision,
- giltighetsintervall och skapandetid,
- strukturerade fakta, aktörer, kampanjer, CVE:er, explicit angivna TTP:er,
  sektorer, regioner, påverkan och osäkerheter,
- `evidence_article_ids` per påstående,
- källantal och oberoende källor,
- prompt-, modell- och schemasignatur,
- varningar från valideringen.

### 4.5 `threat_landscape_reports`

Rapporter lagras separat från vanliga `summary_docs`, men kan speglas till ett
`summary_doc` med `kind: threat_landscape` för befintligt UI. Originalobjektet
innehåller:

```yaml
id: string
profile_id: string
period_start_ts: int
period_end_ts: int
created_at: int
status: draft | published | failed_validation
previous_report_id: string | null
input_snapshot_ids: list[string]
metrics: object
analysis: object
forecast: list[object]
markdown: string
quality: object
prompt_version: string
model: string
```

Spegeln har ett deterministiskt ID, `source_report_id` och en explicit referens
till `threat_landscape_reports` som auktoritativ samling. Den får följa vanlig
`summary_docs`-retention och kan återskapas från originalet vid behov.

Varje analyspåstående refererar kluster-/snapshot-ID:n. Varje prognospost ska ha
`hypothesis`, `horizon`, `confidence` (`low|medium|high`), `rationale`,
`evidence_cluster_ids`, `leading_indicators` och `invalidation_conditions`.
"High" får inte användas innan en senare utvärdering visar att skalan är
kalibrerad; i MVP begränsas utfallet till `low|medium`.

### 4.6 Körningar och vattenmärken

`long_term_runs` sparar typ, profil, start/slut, status, cursor före/efter,
konfigurationshash, statistik och sanerat fel. Vattenmärket är ett stabilt par
`(fetched_at, article_id)`, inte publiceringstid eller enbart en tidpunkt.
Artiklar sorteras på samma par så att sent inkomna artiklar och flera artiklar
med identisk tidsstämpel inte tappas bort. `published_ts` används däremot som
händelsens sorteringstid i klustring och rapportmått. Om en redan behandlad
artikels `content_hash`/embedding-source-hash ändras ska den hamna i en separat
rekonsilieringskö; medlemskapet flyttas aldrig tyst.

Ett lease med timeout hindrar två workerinstanser från att analysera samma profil
samtidigt. En havererad lease kan tas över. Cursor flyttas först när hela batchen
har bestående medlemskap; LLM-sammanfattning får släpa efter och plockas upp via
`membership_revision > summarized_revision`.

## 5. Inkrementell bearbetning

### 5.1 Artikelurval och förkontroll

1. Läs en begränsad batch efter profilens `fetched_at`-vattenmärke och fram till
   ett fast `until_ts` för körningen.
2. Filtrera med profilens källor/kategorier/taggar. Spara orsak och antal för
   bortfiltrerade artiklar. Profilens `categories` är samma feed-ämnen som i
   befintlig konfiguration och ska först lösas till källnamn; artikeldokumentet
   har inget separat kategorifält. `required_tags` matchas mot artikelns taggar.
3. Kräv en giltig `similarity_embedding_vector` med matchande source hash,
   instruktion, modell och dimension. Köa annars omembedding via befintliga
   `ensure_article_embedding()`.
4. Hantera tom text, nollvektor, NaN/Inf och framtida/orimliga tidsstämplar som
   explicita kvalitetsfel; flytta inte cursor förrän de har karantänlagts eller
   hanterats enligt policy.

### 5.2 Händelseklustring

Bearbeta artiklar deterministiskt i ordningen `(article_ts, article_id)`.

1. Hämta endast kluster för samma profil, embedding-signatur och relevant
   tidsfönster. Ett dormant kluster kan återöppnas om matchningen är stark.
2. L2-normalisera vektorer och beräkna cosinuslikhet. Nollvektorer avvisas.
3. Rangordna kandidater med cosinuslikhet och deterministiska skyddssignaler:
   tidsavstånd samt överlapp i starka indikatorer som CVE, namngiven aktör,
   organisation eller produkt. Taggöverlapp får stödja men inte ensam skapa en
   matchning.
4. Matcha alla kompatibla kandidater över likhetströskeln. När kandidaterna
   ligger nära varandra får artikeln medlemskap i flera kluster. Om ingen
   kandidat når tröskeln skapas ett nytt kluster.
5. Skriv varje medlemskap med unik nyckel och uppdatera `vector_sum`, centroid,
   `member_count`, `last_seen_ts` och `membership_revision` atomiskt.

Rena CVE-poster identifieras konservativt: titeln ska ha katalogformat och hela
artikeln ska endast avse samma CVE. Disjunkta CVE-mängder blockerar matchning
bara när både artikeln och kandidatklustret har denna strikta identitet. För
narrativa incident-, kampanj- och rådgivningsartiklar är CVE:er evidens om
händelsen, inte en unik händelsenyckel.

### 5.2.1 Reconciliation av befintliga kluster

`run_long_term_reconciliation.py` läser kluster och medlemskap och producerar
endast förslag. Kandidater måste ha kompatibel embedding-signatur, ligga inom
tidsfönstret och nå review-tröskeln. Auto-förslag kräver dessutom minst `0.88`
likhet och delad stark identitet: samma CVE, eller delad produkt/organisation
tillsammans med ett gemensamt titelbegrepp. Två rena poster med olika CVE:er
blockeras. Narrativ artikel mot ren CVE-post, återkommande katalogpost och
roundup/bulletin går alltid till manuell granskning.

Förslagen binds ihop till granskningsbara komponenter men skrivs inte tillbaka
av proposal-passet. Ett separat apply-kontrakt finns nu för samtliga backends.
Det flyttar medlemskap, räknar om centroid/revision/indikatorer, ger varje
medlemskap en unik sammanhängande revision, bevarar lineage som sekundära
tombstones och nollställer primärens snapshotpekare så att en ny snapshot måste
byggas. Summeringsrevisionen sätts till en syntetisk bas direkt före de nya
medlemsrevisionerna. Äldre snapshots ändras inte och fortsätter därför att
fungera som proveniens för historiska rapporter. SQLite gör övergången i en
transaktion; TinyDB och MongoDB använder idempotenta, återspelningsbara journaler.

CLI-flödet är avsiktligt tvådelat. `--prepare-group` exporterar först en exakt
operationsfil med förväntade revisioner. `--apply-operation` kräver därefter att
hela operations-ID:t upprepas med `--confirm-apply`. Om någon källrevision har
ändrats avvisas operationen utan att SQLite skriver något; dokumentlagrens
journal kan återuppta ett delavbrott till samma deterministiska slutläge.
Superseded tombstones filtreras alltid bort när nya proposals skapas.

Den första godkända apply-körningen genomfördes 2026-09-09 för PaperCut. Sex
kluster blev ett primärkluster med 18 medlemskap och revision 19; fem sekundärer
bevarades som tombstones. Efterkontrollen visade komplett lineage, tom journal
och inga återkommande PaperCut-förslag. Ett fel i den första operationsformen,
som gav alla medlemskap samma slutrevision, upptäcktes innan någon
PaperCut-snapshot sparades. Produktionsmedlemskapen reparerades till unika
revisioner 2–19 och Map har även ett defensivt legacy-skydd. Den resumérbara
cloud-baselinekörningen återstartades därefter.

`0.80` är ett initialt experimentvärde, inte ett produktionskrav. På det
nuvarande materialet har många artiklar minst en granne över denna nivå, men det
säger inget om händelseprecision. Tröskel, tidsgräns och tvetydighetsmarginal ska
kalibreras på ett manuellt märkt artikelpar-/klusterdataset. Målet är i första
hand hög precision och få felaktiga sammanslagningar.

### 5.3 Klusterlivscykel

- `active`: ny evidens inom exempelvis 30 dagar.
- `dormant`: ingen ny evidens inom 30 dagar; visas normalt inte i aktuell
  syntes men historiken bevaras.
- `closed`: analytiskt avslutad händelse eller passerad maximal återöppningstid.

Nya tvetydiga tilldelningar skapar inte `needs_review`-kluster. Den statusen
kan finnas kvar i äldre lagrade data och kan fortfarande öppnas för manuell
hantering. Äldre sådana kluster räknas inte in i nya landskapsmått förrän de
har hanterats.

Tiderna är profilkonfiguration, inte hårdkodade 60 dagar. Livscykeln styr vilka
kluster som tas med i rapporten; den raderar aldrig kluster eller snapshots.

## 6. Map-steg: strukturerad klusteruppdatering

Kör när ett kluster har osammanfattade medlemskap och antingen nått
`min_pending_articles` eller `max_update_delay_hours`. Skicka endast föregående
strukturerade tillstånd och nya artiklar, klippta med befintlig tokenbudgetlogik.

LLM-prompten ska kräva JSON enligt ett versionshanterat schema och instruera:

- använd endast givna artiklar och föregående tillstånd,
- ange artikel-ID:n för varje nytt eller ändrat sakpåstående,
- skilj publiceringstid, händelsetid och rapporterad kampanjperiod,
- bevara kvalificeringar och motstridiga uppgifter,
- ange MITRE ATT&CK-ID endast när källan uttryckligen stöder det,
- markera ersatta/bestridda fakta i stället för att skriva om historiken,
- returnera `insufficient_evidence` när underlaget inte räcker.

Efter anropet ska applikationen:

1. extrahera och JSON-validera svaret,
2. kontrollera att alla evidens-ID:n finns i tillåtet input,
3. kontrollera enumvärden, datum och obligatoriska fält,
4. göra en begränsad reparationsretry vid formatfel,
5. spara snapshot och uppdatera `summarized_revision` med optimistisk låsning,
6. lämna medlemskapen pending och logga ett återkörbart fel om valideringen
   fortfarande misslyckas.

Klusternamn och Markdown renderas från det validerade objektet. De är inte
primära data.

## 7. Reduce-steg: mätning, långtidsanalys och prognos

### 7.1 Deterministiskt analysunderlag

Före LLM-anropet skapar kod, inte modellen, ett kompakt underlag:

- nya oberoende händelsekluster per vecka och profilfacet,
- aktiva, vilande, återöppnade och avslutade kluster,
- unika källor per händelse och källkoncentration,
- rullande fönster, initialt 7, 30 och 90 dagar,
- jämförelse mot föregående motsvarande period och tidigare rapport,
- saknade veckor, ändrad källmix och andra täckningsvarningar.

Räkna händelser, inte artiklar, som primärt trendmått. En trend får normalt
inte deklareras utan flera oberoende händelser över mer än en tidsbucket och
utan att datatäckningen är tillräcklig. Exakta minimikrav konfigureras och
utvärderas; modellen får inte själv hitta på en numerisk utveckling.

### 7.2 Tvåstegssyntes för litet kontextfönster

Om underlaget inte ryms:

1. **Segment-reduce:** sammanfatta validerade snapshots per tidsbucket och
   facet med bibehållna snapshot-ID:n.
2. **Final-reduce:** kombinera segmenten, deterministiska mätvärden och
   föregående rapport till slutligt JSON-objekt.

Rapporten ska innehålla:

- observerade bestående mönster,
- belagda förändringar och strukturella förskjutningar,
- dataluckor och alternativa förklaringar,
- prognoshypoteser för 1–3 månader med evidens, osäkerhet, indikatorer att
  bevaka och villkor som skulle tala emot hypotesen.

Prognosen ska beskrivas som en bedömning, inte ett faktum. Om baslinjen är för
kort, källmixen har ändrats kraftigt eller evidensen är för liten ska systemet
publicera detta i stället för en stark trend eller prognos.

## 8. Konfiguration

Lägg en avstängd standardsektion i `config.yaml.dist`. Exempelvärden ska
kalibreras innan funktionen slås på i produktion:

```yaml
long_term_analysis:
  enabled: false
  ui:
    enabled: false
  schema_version: 1
  batch_size: 200
  quarantine_retry_limit: 50
  # FeedSummary skickar värdet uttryckligen till cleanup; 0 stänger av
  # långtidsretention vid direkt användning av Base-policyn.
  retention_days: 730
  map_lease_seconds: 1800
  report_lease_seconds: 3600
  llm:
    # cloud, local eller configured_chain
    map_provider: cloud
    reduce_provider: cloud
    map_max_output_tokens: 8000
    reduce_max_output_tokens: 2500
  clustering:
    embedding_purpose: similarity
    similarity_threshold: 0.80
    ambiguity_margin: 0.03
    candidate_window_days: 60
    dormant_after_days: 30
    close_after_days: 180
  cluster_updates:
    # Singeltons är atomära observationer och kräver ingen egen LLM-snapshot.
    min_cluster_members_for_map: 2
    min_pending_articles: 3
    max_update_delay_hours: 24
    max_articles_per_call: 12
    article_clip_chars: 2500
  reports:
    mirror_to_summary_docs: true
    windows_days: [7, 30, 90]
    bucket_days: 7
    forecast_horizon_days: [30, 90]
    min_independent_events_for_trend: 3
    source_concentration_threshold: 0.75
    source_mix_change_threshold: 0.35
    min_snapshot_member_count: 2
    max_clusters: 10000
    max_memberships_per_cluster: 10000
    max_snapshots_per_segment: 5
    max_snapshot_records: 10000
    generation_attempts: 3
  profiles:
    healthcare_europe:
      enabled: true
      categories: [Cybersecurity, Dataskydd, Breaches, Sårbarheter]
      priority_tags: [Europe, healthcare]
```

`priority_tags` är prioritering, inte ett obligatoriskt filter, om inte profilen
uttryckligen anger `required_tags`. Konfigurationen ska valideras vid workerstart
och dess normaliserade hash sparas på varje körning och rapport.

Prompter läggs i egna versionshanterade paket under `config/prompts/long_term/`:

- `cluster_update_v1.yaml`
- `cluster_update_v2.yaml`
- `landscape_segment_v1.yaml`
- `landscape_report_v1.yaml`

### 8.1 Framtagande av initial analysprompt

En första version av `landscape_report_v1.yaml` ska tas fram redan i fas 0, innan
Reduce-steget implementeras. De befintliga promptpaketen
`daily_threat_landscape_healthcare_europe_se.yaml` och
`weekly_threat_landscape_healthcare_europe_se.yaml` används som språklig och
analytisk utgångspunkt, men prompten ska anpassas från artikelsammanfattning till
jämförelse av strukturerade klustersnapshots och tidsserier.

Arbetet ska ge följande artefakter:

- en systemprompt som definierar analytikerrollen, källbundenhet,
  osäkerhetshantering och förbudet mot att likställa rapporteringsvolym med
  hotaktivitet,
- en user-template med explicita fält för profil, analysperiod, rullande mått,
  täckningsvarningar, snapshots och föregående rapport,
- ett JSON-schema för observationer, belagda förändringar, alternativa
  förklaringar, dataluckor och prognoshypoteser,
- minst fem representativa prompt-fixtures: normalt underlag, tunt underlag,
  många artiklar om samma händelse, motstridiga källor och ändrad källmix,
- en dokumenterad granskningsrunda där svar bedöms mot evidenstäckning,
  otillåtna slutsatser, formatföljsamhet och stabilitet mellan omkörningar.

Den initiala systemprompten ska minst uttrycka följande kontrakt:

```text
Du är en svenskspråkig strategisk cyberhotanalytiker. Analysera endast det
strukturerade underlag som bifogas. Skilj observerade fakta från analytiska
bedömningar och prognoshypoteser. Varje slutsats ska referera de kluster- eller
snapshot-ID:n som stöder den. Tolka inte fler artiklar eller större
rapporteringsvolym som ökad hotaktivitet. Bevara osäkerheter, motstridiga
uppgifter och alternativa förklaringar. Skapa inte attribueringar, MITRE
ATT&CK-mappningar eller numeriska förändringar som saknas i underlaget. Om
underlaget är otillräckligt ska det framgå i stället för att luckorna fylls med
antaganden. Returnera endast JSON enligt det angivna schemat.
```

Den slutliga v1-prompten ska inte godkännas enbart på språklig kvalitet. Samtliga
fixture-svar måste klara den programmatiska schema- och evidensvalideringen, och
en manuell granskning ska bekräfta att prompten avstår från en trend eller
prognos när stödet är otillräckligt. Modell, temperatur, promptversion och
fixture-version ska sparas tillsammans med utvärderingsresultatet.

## 9. Schemaläggning och drift

Behåll den befintliga timvisa inhämtningen. Lägg till explicita jobbtyper i
`web_viewer/feedsum_worker.py` i stället för att försöka uttrycka analysen som en
vanlig `tag_based_summary`:

```yaml
long_term_incremental:
  frequency: hourly
  type: threat_landscape_incremental
  profile: healthcare_europe

long_term_weekly:
  frequency: weekly
  day: mon
  time: "08:30"
  type: threat_landscape_report
  profile: healthcare_europe
```

Utöka `_entry_to_overrides()` med `profile`. Den inkrementella körningen kan
också kedjas efter lyckad `continuous_tagging`, vilket är robustare än att
förlita sig på YAML-ordning när två timjobb förfaller samtidigt. Rapportjobbet
ska först tömma pending klusteruppdateringar och därefter frysa ett gemensamt
`period_end_ts`.

Workerprocessen kör i dag jobb sekventiellt och har ett globalt
`RUNNING_JOB_ID`. MVP ska respektera detta och använda lagringsleasen för skydd
mot flera processer. Parallella LLM-anrop införs först efter lasttest och med
befintlig rate limiting.

Nödvändig observability:

- antal valda, bortfiltrerade, omembeddade och karantänlagda artiklar,
- nya/matchade/tvetydiga kluster och likhetsfördelning,
- pending/sammanfattade revisioner och dormant/återöppnade kluster,
- LLM-anrop, token-/teckenbudget, retries, svarstid och valideringsfel,
- rapportens evidenstäckning och kvalitetsvarningar,
- cursor-lag i sekunder och äldsta pending medlemskap.

Logga aldrig API-nycklar eller full artikeltext i felmeddelanden.

## 10. Implementationsordning och acceptanskriterier

### Fas 0 – mätbas och testmaterial

- Skapa ett anonymiserat/golden dataset med manuellt bedömda artikelpar och
  händelsekluster, inklusive svåra negativa exempel med samma CVE, aktör eller
  sektor men olika händelser.
- Skapa ett read-only analysverktyg som visar likhetsfördelning, tröskelkurvor
  och föreslagna kluster utan att skriva till databasen.
- Ta fram och versionssätt den initiala analysprompten enligt avsnitt 8.1 samt
  skapa dess fem första prompt-fixtures och granskningsmall.
- Dokumentera startvärden för precision, recall och andel tvetydiga artiklar.

**Klart när:** ett granskat dataset och ett reproducerbart kalibreringskommando
finns; valt tröskelvärde motiveras av mätdata; analyspromptens v1-utkast,
JSON-schema, fixtures och dokumenterade granskningsresultat finns i repot.

### Fas 1 – lagringskontrakt

- Implementera modeller, migrationer/index, repositories och run/cursor-lease.
- Lägg kontraktstester som körs mot TinyDB och SQLite samt MongoDB-integration
  när testdatabas finns.
- Säkerställ att befintliga databaser migreras additivt och att funktionen kan
  vara avstängd utan beteendeförändring.

**Klart när:** samma domänflöde klarar create/read/update, unik medlemsnyckel,
optimistisk revision och lease takeover på samtliga backends. SQLite/MongoDB ska
vara atomiska; TinyDB ska vara processlåst och ha testad journal/reparation när
en flerdokumentskrivning avbryts.

### Fas 2 – deterministisk klustring och backfill

- Implementera vektorvalidering, kandidatval, matchning, centroiduppdatering,
  karantän och livscykel utan LLM-beroende.
- Bygg `--dry-run`, `--limit`, `--profile`, `--since`, `--until` och
  återupptagning för backfill.
- Kör backfill kronologiskt och publicera inte rapporter under backfill.

**Klart när:** samma input kan spelas två gånger utan nya medlemskap eller
centroidförändring; avbruten körning fortsätter utan luckor; golden-datasetets
precisionsmål uppnås.

### Fas 3 – Map-sammanfattning

- Implementera JSON-schema, promptpaket, tokenbudget, formatreparation och
  evidensvalidering.
- Spara snapshots och revisionskoppla dem atomiskt.
- Behåll atomära singeltons i trendmåtten men välj dem inte automatiskt för
  Map. Ett kluster blir snapshot-berättigat när det har minst två medlemmar;
  explicit manuell analys får hanteras separat.
- Lägg golden-/regressionstester med inspelade LLM-svar; enhetstester får inte
  kräva en aktiv Ollama-server.

**Klart när:** inga okända artikel-ID:n kan sparas, ett trasigt LLM-svar skadar
inte klustret och pending arbete kan köras om.

### Fas 4 – mätvärden, Reduce och rapport

- Implementera deterministiska rullande mått och täckningsvarningar.
- Implementera segment-/final-reduce, rapportvalidering, Markdown-rendering och
  jämförelse med föregående rapport.
- Revidera den initiala analysprompten endast utifrån dokumenterade fel i
  fixtures eller validering och höj promptversionen vid kontraktsändringar.
- Spegla publicerade rapporter till `summary_docs` utan att låta ordinarie
  veckogallring radera långtidsoriginalet.

**Klart när:** alla trend- och prognospåståenden har giltiga evidensreferenser,
otillräckligt underlag ger en explicit begränsning och samma frysta input ger
samma deterministiska mått.

### Fas 5 – workerintegration och drift

- Lägg till jobbtyper, profiloverride, status, strukturerad loggning och
  manuella triggerkommandon.
- Testa samtidiga workerstarter, lease-timeout, LLM-timeout, databasfel,
  modellbyte och kontrollerad omkörning.
- Börja med `enabled: false`, kör shadow mode och jämför mot manuella analyser
  innan publicering aktiveras.

Fyra `frequency: triggered`-exempel finns i `config/schedule.yaml.dist`. En
fullständig manuell körning kan, när långtidsfunktionen har aktiverats, startas
via worker-API:t:

```bash
curl -X POST http://127.0.0.1:8799/trigger \
  -H 'Content-Type: application/json' \
  -d '{"name":"long_term_pipeline_healthcare","overrides":{"map_limit":20}}'
```

Tillåtna manuella overrides är `profile`, `since`, `until`, `at`, `limit`,
`map_limit`, `cluster`, `force` och `all_sources`. `all_sources` är avsett för
en uttrycklig initial backfill och kopplar bort profilens källfilter för just
klustringssteget. Övriga fält avvisas.

`limit` är maximalt antal nya artiklar som klustringssteget hämtar efter
profilens cursor i en batch. `map_limit` är maximalt antal kluster som faktiskt
Map-bearbetas i en pass; redan aktuella kluster räknas inte längre mot gränsen.
En vanlig `long_term_pipeline` gör bara en pass i ordningen klustring, Map och
rapport och är därför inte i sig en fullständig historisk backfill.

**Klart när:** en veckas shadow-körningar saknar cursorluckor/dubbletter,
failure injection återhämtas automatiskt och driftmåtten är synliga.

### Fas 6 – API och UI

- Leverera read-only-API och UI bakom `long_term_analysis.ui.enabled`; flaggan
  är avsiktligt oberoende av workerflaggan för att stödja shadow-granskning.
- Visa senaste rapport, tidigare rapporter, period och kvalitetsvarningar.
- Gör kluster-, snapshot- och artikelreferenser klickbara för analytisk
  granskning.
- Visa prognoser som hypoteser med osäkerhet och invalideringsvillkor, tydligt
  separerade från observerade fakta.

**Klart när:** en användare kan följa varje slutsats från rapport till snapshot,
kluster och kvarvarande artikelmetadata utan databasåtkomst.

## 11. Testmatris

Minimikrav före produktionsaktivering:

- **Matematik:** cosinus för normal/ortogonal/nollvektor, dimensionsfel och exakt
  inkrementell centroid.
- **Klustring:** gränsvärde, marginal, tidsgräns, stark indikator, dormant
  återöppning, felaktigt modellbyte och deterministisk ordning.
- **Idempotens:** dubbla event, dubbel körning, avbrott före/efter cursorflytt,
  konkurrerande leases och optimistisk revisionskonflikt.
- **Lagring:** samma kontrakttest mot alla backends, migration från befintligt
  schema samt index/verifierad unik nyckel.
- **LLM-gräns:** ogiltig JSON, okänt evidens-ID, saknat fält, för lång prompt,
  timeout, retry och otillräckligt underlag.
- **Trendlogik:** flera artiklar om en händelse får inte bli flera händelser;
  ändrad källvolym ska ge varning; glest data ska inte ge stark trend.
- **End-to-end:** fixture-artiklar -> medlemskap -> snapshots -> rapport, utan
  nätverk och med fast klocka.
- **Prestanda:** representativt 90-/180-dagarsdataset, minnesgräns,
  kandidatsökning, backfilltid och normal timkostnad.

## 12. Risker och medvetna uppskjutningar

- Onlineklustring är delvis ordningskänslig. Kronologisk stabil ordning och
  golden-tester begränsar detta; periodisk merge/split-rekonsiliering införs
  först efter att felmönster har mätts.
- En centroid kan dölja fler än ett deltema. Kandidatvärdering mot representativa
  medlemsvektorer/medoider är en senare optimering om precisionen kräver det.
- En vektordatabas behövs inte för MVP. Börja med aktiva centroids i minnet och
  inför ANN-index först när mätt latens eller minne motiverar det.
- Automatisk merge/split, analytikerfeedback, sannolikhetskalibrering och
  prognosutfallsbedömning är separata iterationer. Datamodellen ska kunna lagra
  manuella beslut och senare utfall, men MVP ska inte låtsas ha dessa förmågor.
- LLM-modell- eller promptbyte kan ändra texten. Versionerade snapshots och
  fryst input gör skillnaden synlig; gamla rapporter skrivs aldrig över.

## 13. Rekommenderad första leverans

Den första kodleveransen bör omfatta fas 0–2: kalibreringsverktyg,
lagringskontrakt, inkrementell deterministisk klustring och en säker dry-run/
backfill. Därmed kan klusterkvaliteten verifieras innan LLM-sammanfattningar och
prognoser byggs ovanpå dem. Fas 3–5 ska inte aktiveras mot produktionsschemat
förrän felaktiga sammanslagningar, cursorhantering och retention har validerats.
