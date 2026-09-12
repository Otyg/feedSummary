# Singelton-audit 2026-09-09

Auditen är skrivskyddad och avser profilen `healthcare_europe` efter den första
fulla klustringen. Råstatistiken finns i
`long_term_singleton_audit_2026-09-09.json` och 45 par för manuell bedömning i
`long_term_singleton_review_v1.csv`.

## Resultat

- 2 487 artiklar bildade 2 021 kluster.
- 1 814 kluster, 89,8 procent, var singeltons.
- 108 kluster hade två artiklar och 41 hade tre.
- 1 036 singeltons kom från `CVE Latest High/Critical` och 235 från
  `Ransomware.live`. Dessa två atomära källor stod för 70,1 procent av alla
  singeltons.
- Av singeltons som skapades under tröskeln 0,82 hade endast 90 en dåvarande
  bästa match på minst 0,78 och 40 på minst 0,80. En mindre generell
  tröskelsänkning kan därför inte ensam lösa fragmenteringen.
- Det finns samtidigt verkliga splittringar. Flera PaperCut-artiklar med
  parlikhet 0,89–0,94 hamnade separat. CVE-konfliktregeln separerade artiklar
  med olika CVE-mängder och flera senare artiklar blev `needs_review` när två
  redan separata kandidater var nästan lika.
- Hög likhet är inte automatiskt samma händelse: en sammanfattande HPE-artikel
  var mycket lik flera separata CVE-poster om olika sårbarheter. Automatisk
  sammanslagning enbart på en lägre cosinuströskel skulle därför ge falska
  sammanslagningar.

## Beslutad teknisk riktning

Singeltons behålls som atomära observationer i händelse- och trendmåtten men
kräver inte egna LLM-snapshots. Automatisk Map och snapshot-täckningskravet
börjar vid två klustermedlemmar. Det reducerade det aktuella Map-urvalet från
cirka 1 936 kluster till 192 kvarvarande kluster vid verifieringen.

Detta är en kostnads- och täckningspolicy, inte en omklustring. Nästa separata
arbete är att manuellt märka de 45 paren och därefter utforma en försiktig
reconciliation-pass för tydliga multi-source-händelser. Den får inte slå ihop
multi-CVE-bulletiner med varje enskild CVE-post enbart på produkt- eller
textlikhet.
