# Återupptagningsstatus: löpande långtidsanalys

Senast uppdaterad: 2026-09-09

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

- FeedSummaryBase: fullsviten har 227 godkända tester.
- FeedSummary: fullsviten har 148 godkända tester.
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
- Blindat, insert-only JSON-paket för oberoende fulltextgranskning av befintliga
  par; paketet utesluter golden-svar, råa artikel-ID:n och embeddings.
- Versionssatt initial rapportprompt, JSON-schema och fem scenariefixtures.
- Versionssatta Map- och segmentprompter.

Ett första golden-utkast med 33 stratifierade artikelpar, explicit stöd för
`same`/`different`/`ambiguous` och uppmätta precision/recall-värden finns nu.
Utkastet är rubrikgranskat men ännu inte oberoende fulltextgranskat och är
därför inte godkänt golden-material. Dokumenterad granskning av verkliga
modellsvar återstår.
- Golden-utkastet har exporterats till en portabel, anonymiserad float32-fixture
  med 61 artiklar. Den kan köras utan databas eller nätverk. Den faktiska
  onlineklustringen gav precision/recall 0,929/1,000 för `0.80` och
  1,000/1,000 för `0.82`; `0.80` skapade 40 kluster och `0.82` 43. Inget fall
  utlöste `needs_review` bland de produktionshärledda artiklarna. Fixturen innehåller
  därför också två isolerade trepartsscenarier: likvärdiga kandidater ger
  `needs_review` och en motstridig CVE ger korrekt entydig matchning vid båda
  trösklarna. Verkliga fulltextgranskade trepartsfall återstår.
- Fixturen har utökats med 67 frysta, anonymiserade bakgrundskonkurrenter valda
  som de tre närmaste tidigare profilartiklarna inom 60 dagar och över `0.75`.
  En riktad innehållsgranskning korrigerade ett felmärkt AI-par från `same` till
  `different` och lade till tre parvisa relationer för ett verkligt
  CareCloud-trepartsfall. Med bakgrund gav `0.80` precision/recall
  0,929/1,000 och en falsk sammanslagning; `0.82` gav 1,000/1,000 utan fel på de
  28 entydiga relationerna. `0.82` är därmed starkare preliminär kandidat, men
  ännu inte oberoende godkänd.

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

- Evidensbunden `cluster-update-v2` med strikt schema- och ID-validering.
  Version 2 begränsar svarslängden och använder Ollamas strukturerade
  JSON-schemaformat. Version 1 bevaras för historisk reproducerbarhet.
- Deterministiskt urval av pending artiklar och kontrollerad inputbudget.
- Högst ett formatreparationsförsök.
- Atomisk/replaybar lagring av snapshot och `summarized_revision`.
- CLI med preview, limit, force och explicit val av `cloud`, `local` eller hela
  `configured_chain`.
- Driftbeslut 2026-09-09: den aktiva piloten använder `map_provider: cloud`.
  Då filtreras lokala modeller bort även som fallback. Den första lokala
  provkörningen avbröts efter långsam CPU-generering och ogiltigt avklippt JSON;
  ingen snapshot sparades.

### Fas 4 – Reduce och publicering

- Deterministiska 7-/30-/90-dagarsmått på oberoende händelsekluster.
- Jämförelseperioder, veckobuckets, källtäckning, koncentration,
  källmixförskjutning och konservativ `trend_eligible`.
- Direkt eller segmenterad Reduce väljs utifrån kontextbudget.
- Strikt rapportvalidering blockerar okänd evidens, obelagda förändringar och
  prognoser vid otillräckligt underlag.
- När inget mätfönster är trendberättigat tömmer Reduce deterministiskt
  `changes` och `forecast` före validering. Modellens svar kan därför inte
  kringgå den konservativa datagrinden.
- Deterministisk Markdown och rapportidentitet bunden till prompt, modell,
  metrics och snapshots.
- Idempotent originalrapport i `threat_landscape_reports`.
- Valfri deterministisk UI-spegel i `summary_docs`. En sådan spegel är en intern
  extra kopia i FeedSummarys vanliga sammanfattningslista/RSS, inte publicering
  på internet. Konflikter skrivs inte över och ett speglingsfel skadar inte
  originalet. Vanlig summary-cleanup kan radera spegeln men inte originalet.
  Shadow-läget tvingar alltid speglingen av.

### Fas 5 – worker, påbörjad

- Jobbtyperna `long_term_clustering`, `long_term_map`, `long_term_report` och
  `long_term_pipeline` stöds.
- CLI och worker använder samma körfunktioner och samma store-instans.
- Övergripande `long_term_run` innehåller aktuell fas, konfigurationshash,
  kompakt resultat och sanerat fel.
- Blockerad klustring stoppar senare pipelinefaser och lagras som `blocked`.
- Worker-API:t tillåter endast vitlistade långtids-overrides: `profile`,
  `since`, `until`, `at`, `limit`, `map_limit`, `cluster`, `force` och
  `all_sources`. Den sista är endast avsedd för en uttrycklig initial backfill.
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
- En identitetslös artikel når nu den avsedda `blocked`-grenen i stället för att
  filtreras bort och orsaka cursorfel. Efter korrigering kan en ny körning
  fortsätta från oflyttad cursor. Workerhistoriken behåller separata
  `blocked`/`failed`-körningar när en senare körning lyckas. Misslyckade Map- och
  Reduce-svar kan köras om utan dubbla insert-only-dokument. Byte av
  embeddingmodell, dimension eller instruktion hålls strikt i separata kluster.
- Ett deterministiskt och nätverksfritt kapacitetsbenchmark täcker 90 och 180
  dagar med fyra syntetiska händelser per dag, 64-dimensionella embeddings och
  100 kandidatprober. Standardkörningen 2026-09-08 gav 2,920 s/3,054 MB för 90
  dagar och 5,650 s/5,330 MB för 180 dagar. Båda låg under gränserna 15 s och
  512 MB. Kandidaturval, mått, snapshoturval och Reduce-input mäts separat via
  `benchmark_long_term_analysis.py`.
- `long_term_analysis.retention_days` förs nu uttryckligen in i cleanup-policyn.
  TinyDB, SQLite och MongoDB gallrar utgångna rapporter, orefererade snapshots,
  terminala körningar och lösta karantänposter. Endast utgångna stängda kluster
  gallras, tillsammans med sina medlemskap; snapshots som refereras av en
  kvarvarande rapport skyddar hela klustrets evidenskedja. Base har värdet `0`
  som säker opt-in-standard, medan appens distributionskonfiguration anger 730
  dagar. Kontraktstestet kör samma retentionfall mot alla tre backends och
  workern loggar borttagningsräknare per lagringstyp.
- `long_term_analysis.mode` är nu explicit `shadow` eller `publish`. I
  `shadow` tvingar rapportkörningen `mirror_to_summary_docs=false` oavsett
  speglingsflaggan, och körningsposten lagrar valt läge.

### Aktiv shadow-pilot 2026-09-08

- Den lokala, git-ignorerade `config.yaml` har `enabled: true`, `mode: shadow`,
  UI aktiverat, preliminär likhetströskel `0.82`, klusterbatch 25 och
  `map_provider: cloud` samt `reduce_provider: cloud`. Den lokala modellen kan
  därför inte användas av Map eller Reduce. `config/schedule.yaml` har fyra
  manuella pilotjobb plus klustring varje timme, Map var sjätte timme och
  shadow-rapport måndag 05:30.
- Den senaste skrivande klusterbatchen behandlade 25 artiklar: 24 nya kluster,
  en matchning med indikatorstöd, ingen karantän, inget `needs_review` och ingen
  blockerare. Tillsammans med tidigare pilotdata finns 50 medlemskap i 49
  kluster. Körningen `long_term_run_74f44f5285d8459dbcd1fa286970645c`
  är `done` och cursorn är satt.
- Ett första lokalt Map-försök med `gemma4:e2b` gav långsam CPU-generering och
  avklippt JSON. Körningen avbröts under formatreparation och sparade ingen
  snapshot.
- Efter uttryckligt godkännande att skicka det publika artikelinnehållet till
  cloud-kedjan skapades fem av fem snapshots på cirka 21 sekunder. Alla
  validerades utan formatreparation.
- En större återupptagningsbar Map-körning sparade därefter 23 snapshots innan
  ett ogiltigt cloud-svar stoppades av valideraren. Totalt finns 28 snapshots.
  De återstående 21 singelklustren var ännu inte Map-berättigade vid omkörning
  och tas av kvartdygnsjobbet efter 24-timmarsgränsen.
- En efterföljande read-only rapportpreview publicerade inget. Den såg 49
  händelsekluster, 28 snapshots och 21 saknade snapshots samt varningarna
  `insufficient_time_buckets`, `missing_comparison_events`,
  `missing_comparison_sources`, `missing_time_buckets`,
  `no_independent_events`, `no_known_sources` och
  `pending_cluster_snapshots`.
- En verklig shadow-rapport skapades 2026-09-08/09 från 28 snapshots i 6 segment
  och 7 cloud-anrop, utan formatreparation. Rapport-ID är
  `threat_report_a9d3f30b82325e5791013db138d16ff9`; slutprompten var uppskattat
  15 086 av 20 476 tillgängliga tokens och `mirror_action` var `disabled`.
- Worker och båda UI-varianterna verifierades via HTTP (`/health` respektive
  `/long-term`, samtliga HTTP 200). De stoppades därefter på användarens begäran
  för användarstyrd omstart; vid handoff var portarna 8799, 5000 och 5001 fria.
  Löpande extern cloud-behandling av det publika underlaget är uttryckligen
  godkänd för shadow-provet.

### Initial full backfill 2026-09-09

- `limit` begränsar nya artiklar efter profilens cursor i en klustringsbatch.
  `map_limit` begränsar kluster som faktiskt Map-bearbetas i en pass. Den
  vanliga `long_term_pipeline` kör exakt en pass av klustring -> Map -> rapport
  och tömmer inte automatiskt en historisk backlog.
- Klustringen kördes därför separat med `--limit 5000 --all-sources`. Den
  initiala databasen växte under arbetet och slutläget blev 2 487 artiklar,
  samtliga med medlemskap. Profilen har 2 021 kluster: 2 001 aktiva och 20
  `needs_review`. Ingen artikel hamnade i karantän.
- Map-backloggen efter klustringen var 1 974 aktiva kluster och minst 1 975
  modellkörningar med `max_articles_per_call: 12`. Ett verifieringsanrop mot
  cloud-kedjan lyckades. En första stor batch avbröts därefter av ogiltig JSON
  från ett enskilt kluster efter två sparade snapshots.
- Map-CLI:t kan nu isolera klusterfel med `--continue-on-error`, och urvalet
  skannar förbi redan aktuella kluster innan `map_limit` appliceras. Den nya
  `run_long_term_backfill.py` upprepar Map, verifierar att backloggen är tom och
  kör först därefter rapporten.
- Den första feltoleranta återupptagningen startades som user-systemd-enheten
  `feedsummary-long-term-baseline.service` med cloud Map/Reduce, men stoppades
  uttryckligen 2026-09-09 när klusterfördelningen visade kraftig fragmentering.
  Den hann skapa totalt 308 snapshots; ingen ny rapport skapades. Fördelningen
  var 1 814 singeltons (89,8 procent av klustren), 108 dubbeltons och i
  genomsnitt 1,23 artiklar per kluster.
- Efter singelton-policy och PaperCut-reconciliation återstartades samma
  resumérbara baseline 2026-09-09 14:02 CEST med invocation-ID
  `7ee2d9dbf6e949a28eebf5bb8dd2b1b9`, `--skip-clustering`, `--map-limit 2000`
  och `--max-map-passes 4`. Enheten använder konfigurerad cloud Map/Reduce.
  Kontrollera enheten med:

  ```bash
  systemctl --user status feedsummary-long-term-baseline.service
  journalctl --user-unit feedsummary-long-term-baseline.service -f
  ```
- Den körningen visade att 22 av 196 kluster lyckades medan 174 misslyckades,
  men den äldre progressloggen visade bara kluster-ID:n. Körningen stoppades
  innan fyra kostsamma retry-pass slutfördes. Rotorsaken verifierades med
  isolerade cloud-anrop: Map-svaren klipptes mitt i JSON vid både 1 200 och
  2 500 outputtokens, och reparationssvaret maskerade originalsfelet.
- `cluster-update-v2` begränsar nu svarets poster/text, skickar hela JSON-schemat
  via Ollamas strukturerade `format` och använder 5 000 outputtokens. Ett
  skrivande produktionstest av det tidigare felande Plex-klustret sparade
  revision 4 utan reparationsförsök. Map loggar nu `started` samt
  `saved`/`failed` för varje försök med passnummer, position, feltyp och sanerat
  felmeddelande. Båda valideringsfelen och svarslängderna bevaras om även
  formatreparationen misslyckas.
- Baseline återstartades därefter med invocation-ID
  `269a79b354c1471d915581745f079be6`. Den startade med 171 kvarvarande kluster;
  tidigare sparade snapshots återanvändes.
- Körningen avslutades efter fyra Map-pass med exitstatus 2 eftersom backloggen
  inte blev helt tom. Den skapade ingen ny rapport. Slutläget vid kontroll var
  502 snapshots, 21 valbara kluster och 51 väntande revisionsenheter. Ingen
  baseline-process eller aktiv lease fanns kvar. Senaste rapporten var fortsatt
  `threat_report_a9d3f30b82325e5791013db138d16ff9`.
- Fyra kluster misslyckades återkommande i sista passet:
  `threat_cluster_e8d1ba52596b5b0997737d9f264b0245`,
  `threat_cluster_5a171e21f32d5925a832977efb03fce0`,
  `threat_cluster_b958b89763d257378397f2803accff28` och
  `threat_cluster_2f7116f0f72d50fb9a28da51ebe02692`. Tre fel berodde på
  en tom sträng i `uncertainties`; det fjärde svaret klipptes fortfarande mitt
  i JSON vid 5 000 outputtokens. PaperCut-klustret lyckades däremot i två
  avgränsade batcher och är fullständigt summerat till revision 19.
- De övriga 17 av de 21 väntande klustren tillkom eller fick nya medlemskap via
  det timvisa jobbet `long_term_cluster_healthcare_shadow` medan baseline
  kördes. Detta är normal löpande ingestion, inte kvarvarande fel från den
  ursprungliga Map-backloggen.
- Slutförandet 2026-09-10 höjde endast Maps outputreserv till 8 000 tokens.
  Det tidigare trunkerade klustret
  `threat_cluster_b958b89763d257378397f2803accff28` sparades då till revision
  5 utan reparation. Tomma/null uncertainty-markörer tas bort evidensneutralt;
  strukturerade objekt bevaras förlustfritt som deterministisk JSON-text. Även
  de tre tidigare uncertainty-felen kunde därefter sparas.
- En ny resumérbar Map-pass med invocation-ID
  `2d69854e127f45189a8c8de750a5758f` sparade samtliga 19 då återstående
  kluster med 0 fel. Reduce behövde därefter förstärkas separat: JSON-schema
  skickas nu till alla anrop, segmentprogress loggas och ett transient fel gör
  upp till tre lokala genereringsförsök utan att tidigare segment behöver
  upprepas.
- Reduce använder fortsatt 2 500 outputtokens för att lämna 20 476 tokens åt
  input. Promptens måttvy behåller aggregat men inverterar upprepade
  tidsbucket-ID:n per citerbart kluster och tar bort icke-citerbara singleton-ID:n
  samt per-händelse-källkartor. Originalmåtten används oförändrade vid strikt
  validering och persistens. Om segmenttexten ändå är för stor begränsas
  detaljnivån jämnt och deterministiskt utan att något citerbart kluster-ID tas
  bort.
- Slutkörningen med invocation-ID `e605e551ffca4e5b9bd1c3dd7962cf7b`
  validerade 44 segment. Segment 35 återhämtades lokalt på försök 2. Finalen
  rymdes redan på komprimeringsnivå 1 med 15 720 av 20 476 inputtokens och
  sparade `threat_report_72ea6f4fdae356c49a0bb82bbd926dff` från 219
  snapshots. Enheten avslutade med status 0.
- Rapporten är `published` i det separata långtidslagret men kördes i
  `mode: shadow`; `summary_threat_report_72ea6f4fdae356c49a0bb82bbd926dff`
  finns därför inte. Rapporten har en observation, inga changes/prognoser och
  kvalitetsvarningarna `missing_comparison_events`, `missing_comparison_sources`,
  `missing_report_snapshots`, `missing_time_buckets` och
  `pending_cluster_snapshots`. De två poster som hann tillkomma efter den frysta
  rapporttidpunkten Map-behandlades direkt efteråt. Slutlig tvingad preview gav
  `eligible_clusters: []`. Den körande UI-processen svarade HTTP 200 för
  rapportvyn, och markup-kontrollen visade mänskliga evidensetiketter som
  `CVE-2026-84226 – OpenVPN Security Vulnerabilities September 2026` medan
  interna kluster-ID:n endast låg i URL och tooltip.

### Singelton-audit och Map-policy 2026-09-09

- Den stora Map-backfillen stoppades efter 308 sparade snapshots och utan ny
  rapport. Den föräldralösa profilleasen frigjordes.
- Den reproducerbara auditen ligger i `audit_long_term_singletons.py`.
  Statistik och 45 granskningspar finns i
  `long_term_singleton_audit_2026-09-09.json` respektive
  `long_term_singleton_review_v1.csv`; slutsatserna är sammanfattade i
  `long_term_singleton_findings_2026-09-09.md`.
- 1 814 av 2 021 kluster var singeltons. 70,1 procent av singeltonerna kom från
  de atomära källorna `CVE Latest High/Critical` och `Ransomware.live`.
- `cluster_updates.min_cluster_members_for_map: 2` hindrar nu automatiska
  LLM-anrop för singeltons. `reports.min_snapshot_member_count: 2` behåller
  dem i trendmåtten men undantar dem från snapshot-backlog och
  rapporttäckningskrav. Default i Base är fortsatt 1 för bakåtkompatibilitet.
- Read-only-verifieringen valde cirka 192 kvarvarande Map-kluster i stället för
  cirka 1 936. Policyn godkändes för en initial shadow-baseline även om
  parametrarna ännu inte är slutkalibrerade.
- Online-klustringen skiljer nu rena en-CVE-katalogposter från narrativa
  händelseartiklar. Disjunkta CVE:er är endast ett hårt stopp när båda sidor är
  rena CVE-poster; klustret tappar denna strikta identitet så snart en narrativ
  medlem tillkommer. Flaggan sparas på både medlemskap och kluster, medan gamla
  kluster klassificeras från bevarad medlemsevidens i reconciliation.
- `run_long_term_reconciliation.py` och Base-modulen
  `long_term/reconciliation.py` implementerar ett separat read-only
  proposal-pass. Det kräver semantisk likhet plus händelseidentitet för
  auto-förslag, blockerar två skilda rena CVE-poster och skickar ren-CVE mot
  narrativ samt roundup/bulletin till review.
- Verifiering 2026-09-09 mot 2 027 kluster: 159 393 kandidater jämfördes, 17
  auto-kanter gav tre grupper med totalt tio kluster, 265 par gick till review
  och 1 863 hög-lika par blockerades som skilda rena CVE-poster. PaperCut blev
  en enda grupp om sex kluster. CISA:s återkommande katalogposter och en weekly
  roundup blev inte auto-grupper.
- Apply-kontraktet är nu implementerat i TinyDB, SQLite och MongoDB. Det flyttar
  medlemskap till en vald primär, räknar om vektorsumma/centroid/revision och
  indikatorer, tilldelar varje flyttat medlemskap en unik sammanhängande
  revision, nollställer primärens snapshotpekare och lämnar sekundärerna som
  stängda tombstones med `superseded_by_cluster_id`. Äldre snapshots bevaras
  oförändrade. Retention skyddar både applicerad lineage och pending journaler.
- SQLite använder en enda transaktion. TinyDB och MongoDB använder en
  idempotent reconciliation-journal; failure-injection verifierar återstart
  efter att klusterdokumenten skrivits men innan medlemskapen flyttats. Samma
  operation kan köras om utan dubbel flytt, medan stale revision avvisas.
- CLI:t har två avsiktligt separata steg: `--prepare-group` skriver en ny,
  granskningsbar operationsfil och `--apply-operation` kräver hela ID:t via
  `--confirm-apply`. PaperCut-gruppen förbereddes i
  `/tmp/papercut_reconciliation_operation_v1.json`: sex källkluster, 18
  medlemskap, målrevision 19 och operations-ID
  `threat_reconciliation_6c4ccb7d0f5de8ad084cb0a9d86edc28`.
- PaperCut-operationen applicerades 2026-09-09 efter uttryckligt godkännande.
  Den äldre operationsformen gav först samtliga 18 medlemskap slutrevision 19.
  Baskörningen stoppades innan någon PaperCut-snapshot sparades. Koden allokerar
  nu unika revisioner för nya operationer och Map har ett defensivt legacy-skydd.
  Produktionsklustret reparerades villkorsstyrt till medlemsrevisionerna 2–19
  och `summarized_revision: 1`; preflighten visade exakt 18 väntande artiklar.
  Alla medlemskap har lineage, fem sekundärkluster är stängda tombstones och
  reconciliation-journalen är tom.
- Proposal-passet filtrerar nu uttryckligen bort superseded tombstones. En ny
  körning efter apply innehöll inga PaperCut-grupper eller PaperCut-auto-kanter.
  Eftersom ordinarie ingestion/klustring fortsatte parallellt omfattade snittet
  2 049 icke-superseded kluster och gav fyra andra auto-grupper; ingen av dem
  har applicerats.
- UI:t visar en tydlig lineage-länk från en tombstone till det sammanslagna
  klustret och förklarar att äldre snapshots är historisk proveniens.
- Baseline-enheten är inte längre aktiv. Nästa operativa steg är manuell
  innehållsgranskning av den sparade shadow-rapporten och ett stratifierat urval
  av dess snapshots mot originalartiklarna. Följ därefter nästa schemalagda
  pipeline för att verifiera att de två nytillkomna snapshotsen tas med och att
  progress/retries fungerar utan handpåläggning. De fyra kvarvarande
  auto-grupperna ska fortsatt granskas manuellt; applicera endast namngivna,
  godkända operationer.

  Efter godkänd granskning kan samma säkra återupptagning startas igen; redan
  sparade snapshots hoppas över:

  ```bash
  systemd-run --user --unit=feedsummary-long-term-baseline --collect \
    --property=WorkingDirectory=/home/otyg/projects/feedSummary \
    /home/otyg/projects/feedSummary/.venv/bin/python \
    run_long_term_backfill.py --config config.yaml \
    --profile healthcare_europe --skip-clustering \
    --map-limit 2000 --max-map-passes 4
  ```

### Fas 6 – API och UI, påbörjad

- Beslut 2026-09-08: UI-arbetet går parallellt med produktionsvalideringen och
  är inte beroende av att workerpipen aktiveras.
- Den separata flaggan `long_term_analysis.ui.enabled` styr all åtkomst till de
  nya läs-API:erna och vyerna. Den kan visa sparade shadow-rapporter medan
  `long_term_analysis.enabled` fortsatt är `false`.
- `GET /api/v1/long-term/reports` har ett versionerat, kompakt listkontrakt med
  profilval. `GET /api/v1/long-term/reports/<report_id>` returnerar den
  auktoritativa rapporten.
- `/long-term` och `/long-term/report/<report_id>` visar historik, tomläge,
  period, kvalitetsvarningar och proveniens-ID:n. Prognoshypoteser markeras som
  icke observerade fakta och visar osäkerhet, indikatorer och
  invalideringsvillkor.
- Läs-API och vyer för snapshot och kluster gör nu hela kedjan rapport ->
  snapshot -> kluster -> artikel klickbar. Kluster-API:t filtrerar bort centroid
  och vektorsumma. Om en artikel har gallrats visas medlemskapets bevarade
  evidensmetadata i stället för en trasig intern artikellänk.
- Evidenslänkar visar nu mänskliga etiketter från den frysta snapshottiteln och
  kompletterar med belagd CVE när titeln saknar den, exempelvis
  `CVE-2026-65974 – ERPNext Server-Side Template Injection`. Artikellänkar
  använder aktuell eller bevarad rubrik. Interna ID:n behålls endast i URL,
  tooltip och teknisk proveniens.
- Base-kontraktet har fått direktuppslag via `get_cluster_snapshot()` i TinyDB,
  SQLite och MongoDB; samma kontraktstest är godkänt för samtliga backends.
- `/long-term/clusters` och `GET /api/v1/long-term/clusters` ger nu en
  paginerad översikt med profil- och statusfilter. Varje rad länkar vidare med
  en mänsklig snapshot-/evidensrubrik och visar medlemsantal, senaste
  observation samt väntande revisionsenheter. Lagringskontraktet stöder offset
  och projektion utan centroid/vektorsumma i TinyDB, SQLite och MongoDB. Både
  normal- och read-only-viewern exponerar översikten.
- `/long-term/review/<cluster_id>` visar nu den tvetydiga artikeln mot bästa och
  näst bästa kandidat, med poäng, indikatorer och bevarad artikelevidens. Nya
  medlemskap sparar båda kandidat-ID:n. För de 26 äldre `needs_review`-
  singeltonerna rekonstrueras kandidaterna vid läsning och UI:t varnar för att
  klustren kan ha förändrats sedan tilldelningen.
- Den skrivbara viewern har ett revisionskontrollerat POST-kontrakt för
  `keep_separate`, `exclude` och `merge`. Icke-merge-beslut är atomiska i alla
  tre lagringsbackendar; merge använder befintligt reconciliation-kontrakt.
  Granskare, tid, motivering och eventuellt målkluster bevaras. Åtgärderna
  kräver `long_term_analysis.ui.review_actions_enabled: true`, medan read-only-
  viewern alltid svarar `401` på skrivförsök. Flaggan lämnas avstängd tills den
  skrivbara interna viewerns åtkomst har verifierats.
- Nya landskapsmått ignorerar både olösta `needs_review`-kluster och explicit
  exkluderade kluster och redovisar antalet i kvalitetsmetadata.
- Klusterlistan kan nu filtreras på minsta antal artiklar före paginering.
  Klusterdetaljen har ett skrivskyddat arbetsflöde för att flytta en artikel
  till ett befintligt kluster eller bryta ut den till ett nytt singletonkluster.
  Ändringen är atomisk och revisionskontrollerad i SQLite samt journalförd och
  återupptagningsbar i TinyDB/MongoDB. Granskare, tid och motivering sparas;
  historiska snapshots bevaras medan berörda kluster markeras för full
  ombyggnad i nästa Map-pass. Read-only-viewern avvisar endpointen explicit.
- Nästa UI-steg är paginering/sökning bland kluster över 500 medlemskap samt
  slutlig åtkomst-, navigations- och deploymentgranskning i read-only-viewern.

## Blockerare före produktionstest

### 1. Produktionsdata och klusterkalibrering

- Kör embedding-audit mot hela produktionsprofilen.
- Gör en begränsad och säker `similarity`-backfill för saknade embeddings.
- Fulltextgranska golden-utkastets 33 par oberoende och utöka det med fler
  svåra negativa samt låg-likhetspositiva exempel.
- Verifiera den riktade granskningen oberoende: vårdparet är fortsatt
  `different`, medan AI-paret korrigerades till `different`. CareCloud-fallet
  ger ett verkligt positivt trepartskluster. Den isolerade tvetydighets- och
  CVE-logiken är verifierad. Med bakgrund gav `0.80` precision 0,929 och `0.82`
  precision 1,000; båda hade recall 1,000. Varken tröskel är ännu godkänd.
- Granska karantän, felaktiga sammanslagningar och splittrade händelser efter en
  liten skrivande klusterbatch.

### 2. Verkliga LLM-regressionsfall

- Kör Map-, segment- och rapportprompterna mot exakt de modeller som ska användas.
- Spara anonymiserade svar som fixtures.
- Verifiera JSON-följsamhet, truncering, evidensreferenser, svenska,
  motstridiga källor och tunt underlag.
- Justera prompt först efter dokumenterat fel och höj promptversionen vid en
  kontraktsändring.

### 3. Paketering och databasberedskap

- Committa endast avsedda filer i båda repona.
- Publicera/versionera Base och lås appen till den versionen eller en exakt
  commit.
- Ta backup av produktionsdatabasen.
- Verifiera additiv schemainitialisering, unika index och store-kontrakt mot den
  verkliga MongoDB-versionen, inte bara mongomock.
- Lägg långtidsjobben i den faktiska produktions-schedulefilen; de finns ännu
  bara i distributionsmallen.

### 4. Shadow-konfiguration och driftinsyn

- Använd `mode: shadow` och behåll dessutom
  `reports.mirror_to_summary_docs: false` som försvar på djupet. Håll
  `long_term_analysis.ui.enabled: false` tills en avsedd granskare ska få
  åtkomst till de nya read-only-vyerna; den aktiva lokala piloten har flaggan
  påslagen för denna granskning.
- Håll worker-API:t avstängt eller bundet till loopback. Det har ingen egen
  autentisering och får inte exponeras externt.
- Lägg minsta nödvändiga status/alerting för `failed`, `blocked`, lease-konflikt,
  karantänstorlek, snapshot-lag, cursor och LLM-körtid.
- Verifiera cleanup-räkningarna mot en databasbackup under shadow-testet innan
  retention tillåts gallra verklig långtidsdata.

## Rekommenderad återupptagningsordning

1. Låt en oberoende granskare kontrollera de 33 paren, lägg fler svåra positiva
   trepartsfall och besluta om `0.82` ska användas i en begränsad shadow-körning.
2. Kör produktionsdatans embedding-audit och en liten kontrollerad backfill.
3. Spela in verkliga Map-/Reduce-svar och lägg regressionsfixtures.
4. Versionera Base, lås appens beroende och verifiera deployment mot en kopia av
   produktionsdatabasen.
5. Kör shadow i ordningen klustring, manuell granskning, Map, manuell
   granskning, rapport utan spegling och först därefter full pipeline.
6. Låt shadow-körningen gå minst en vecka och godkänn först när den saknar
    cursorluckor/dubbletter och de analytiska resultaten har granskats manuellt.

## Första konkreta nästa arbetsuppgift

1. Verifiera de rekonstruerade kandidaterna för ett stratifierat urval av de 26
   äldre `needs_review`-klustren. Kontrollera därefter åtkomsten till den
   skrivbara interna viewern och sätt först då
   `long_term_analysis.ui.review_actions_enabled: true` för en liten pilot.
2. Granska `threat_report_72ea6f4fdae356c49a0bb82bbd926dff` manuellt i UI:t
   och kontrollera dess enda observation, dataluckor och evidenslänkar mot ett
   stratifierat urval av originalartiklar.
3. Följ nästa schemalagda Map/rapportkörning. Bekräfta att normal ingestion inte
   lämnar en växande backlog och att Reduce visar segmentprogress samt lokala
   retries vid behov.
4. Spela in avidentifierade verkliga Map-/Reduce-felsvar som regressionsfixtures,
   särskilt strukturerade uncertainties, dubbla JSON-objekt och avklippta svar.
5. Fortsätt oberoende granskning av golden-paketets 33 par och de fyra ännu inte
   applicerade reconciliation-grupperna innan klustertröskeln godkänns.

Behåll `mode: shadow` och `reports.mirror_to_summary_docs: false`. Timvis
klustring kan skapa nya väntande kluster medan backfillen kör; skilj dessa från
de fyra dokumenterade formatfelen. Den oberoende granskningen av golden-paketets
33 par kvarstår och `0.82` är fortfarande preliminär.

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

## Go/no-go för fortsatt shadow-test och senare publicering

Den begränsade shadow-körningen har startats efter uttryckligt beslut att börja
även om parametrarna ännu inte är slutkalibrerade. Följande krav återstår innan
bred produktionsaktivering eller `mode: publish`:

- Base-versionen är entydigt installerad,
- databasbackup och indexverifiering är klara,
- outputgräns och lease-livslängd är säkrade,
- en liten embedding-backfill och klustergranskning är godkända,
- riktiga LLM-svar passerar validering utan okänd evidens,
- grundläggande failure-injection är grön,
- spegling till `summary_docs` är avstängd,
- status och larm för blockerade/felande körningar finns.
- retentionens borttagningsräkningar har verifierats mot en databasbackup.

Fas 6, med fullständigt API/UI och klickbar evidens, är inte ett krav för
shadow-test men krävs före bred användarpublicering.
