# Eurostat Agent — Stage 11 evidence audit

Source: uploaded `stage11_robustness.json` (schema 1.0, 36 cases per model). This is a retrospective analysis of the saved run, not a rerun or independent verification against live Eurostat.

## Result overview

| Indicator | Llama 3.2 3B constrained | Qwen 3.5 constrained |
|---|---:|---:|
| Supported exact | 11 | 13 |
| Supported cases | 20 | 20 |
| All completed (including partial unsupported) | 14 | 13 |
| Safe-failure labels | 5 | 6 |
| Architecture limitation labels | 3 | 0 |
| Unexpected failures | 9 | 3 |
| Recorded provenance violations | 0 | 0 |
| Model calls | 68 | 56 |
| Cache hits | 57 | 34 |
| Cache misses | 117 | 74 |

## Principal findings

1. **Request coverage:** Llama E02/E03/E05 returned a single verified observation from multi-value requests. Their plans dropped a requested country or sex. The original `completed_inexact` records should be preserved, then additionally reported as `unsupported_request_partially_answered`; the returned values must not be counted as successful complete answers.
2. **Diagnosed cause versus stopping stage:** Every `unexpected_failure` is a Eurostat HTTP 400 at retrieval, but many were preceded by incorrect dataset planning/selection (e.g., A03, B03, B05, B06, D01/D05) or a malformed time interval (C05). The HTTP status alone does not identify the root cause.
3. **Temporal:** Llama D01/D02 used correct annual `2024` yet failed on a dataset lacking the requested sex dimension. Llama D03/D04/D06 used `2024-01-01`; D04 failed earlier on the dataset's dimensions. Qwen D03 used annual `2024` but failed to resolve `women` in the chosen dataset; Qwen D06 produced no plan. Separate temporal errors from dataset/code errors.
4. **Model output contract:** Qwen has 15 planner-call failures in total: 12 labeled model-completion failure and 3 labeled safe failure. All are invalid JSON or invalid field types; this is not evidence of deliberate refusal. Raw model responses were not retained, limiting diagnosis.
5. **Safe failures:** Llama F03/F05 and Qwen F03 fail for a different reason than the actual unsupported subject or operation. Qwen F04–F06 fail at invalid JSON. The reported safe-failure rate means no answer was emitted, not that the system recognized why a request was unsafe or underspecified.
6. **Score denominators:** The three displayed accuracy metrics of 1.0 are calculated over only 11 Llama / 13 Qwen scored, supported, completed cases; they do not include all 36 prompts or the Llama partial answers.
7. **Provenance:** All 27 emitted answer objects carry source provenance and computation objects; 0 recorded violation flags. This does not independently verify source values, the completeness of the validator, or semantic coverage of the original question.
8. **Performance:** Llama controller total median 2.24 s and Qwen 514.10 s; Qwen's planner-call median is 430.91 s. These are local-run observations, not general hardware-independent model performance.
9. **Cache:** Llama 57 hits/117 misses, Qwen 34/74. Execution-scoped cache misses recur between cases; do not describe these as a cross-case hit rate or a demonstration of overall speedup without an uncached control.

## Suggested disposition

**Keep the original run intact.** Add a post-hoc diagnostic annotation or companion report rather than retroactively rewriting observations. Before merging Stage 11, consider a minimal reporting correction to distinguish partial unsupported answers, incidental safe failures, scored-case denominators, and diagnosed cause versus observed stage. Do not redesign `QuestionPlan` or normalize dates within Stage 11.

## Side-by-side outcome matrix

| Case | Expected capability | Llama outcome | Qwen outcome | Llama selected dataset | Qwen selected dataset |
|---|---|---|---|---|---|
| A01 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| A02 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| A03 | supported | unexpected_failure | model_completion_failure | urt_lfe3emp | — |
| A04 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| A05 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| A06 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| B01 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| B02 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| B03 | supported | unexpected_failure | completed_exact | urt_lfe3emp | demo_pjan |
| B04 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| B05 | supported | unexpected_failure | completed_exact | ilc_atsb01a | demo_pjan |
| B06 | supported | unexpected_failure | unexpected_failure | demo_pjangroup | cens_hnmga |
| C01 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| C02 | supported | completed_exact | completed_exact | demo_pjan | demo_pjan |
| C03 | known_unsupported | unexpected_failure | model_completion_failure | demo_pjanbroad | — |
| C04 | known_unsupported | expected_architecture_limitation | model_completion_failure | demo_pjan | — |
| C05 | known_unsupported | unexpected_failure | model_completion_failure | demo_pjan | — |
| C06 | known_unsupported | expected_architecture_limitation | model_completion_failure | demo_pjan | — |
| D01 | supported | deterministic_validation_failure | unexpected_failure | tesem060 | cens_hnmga |
| D02 | supported | deterministic_validation_failure | deterministic_validation_failure | tesem060 | cens_hnmga |
| D03 | supported | unexpected_failure | deterministic_validation_failure | tesem010 | cens_hnmga |
| D04 | supported | deterministic_validation_failure | completed_exact | tesem060 | demo_pjan |
| D05 | supported | completed_exact | unexpected_failure | demo_pjan | cens_hnmga |
| D06 | supported | unexpected_failure | model_completion_failure | urt_lfe3emp | — |
| E01 | known_unsupported | model_completion_failure | model_completion_failure | — | — |
| E02 | known_unsupported | completed_inexact | model_completion_failure | demo_pjan | — |
| E03 | known_unsupported | completed_inexact | model_completion_failure | demo_pjan | — |
| E04 | known_unsupported | expected_architecture_limitation | model_completion_failure | demo_poppcctz | — |
| E05 | known_unsupported | completed_inexact | model_completion_failure | demo_pjan | — |
| E06 | known_unsupported | model_completion_failure | model_completion_failure | — | — |
| F01 | safe_failure | safe_failure | safe_failure | tps00001 | tps00001 |
| F02 | safe_failure | safe_failure | safe_failure | — | — |
| F03 | safe_failure | safe_failure | safe_failure | tps00001 | tps00001 |
| F04 | safe_failure | unexpected_failure | safe_failure | ei_lmhr_m | — |
| F05 | safe_failure | safe_failure | safe_failure | ilc_hch05b | — |
| F06 | safe_failure | safe_failure | safe_failure | — | — |

## Case-level evidence index

Each entry gives the preserved question, plan, selected dataset, answer value or failure. The full source JSON retains the unabridged trace, timestamps, metadata and statistical provenance.

### llama32-3b-constrained

**A01 — completed_exact** (`supported`)
- Question: What was the population of 20-year-old women in Belgium in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**A02 — completed_exact** (`supported`)
- Question: What was the population of 20-year-old men in Germany in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=DE, sex=male; op=none`
- Selected: `demo_pjan`; value: `419725.0`; stopped at: `—`.
- Error: —

**A03 — unexpected_failure** (`supported`)
- Question: How many 30-year-old women lived in France in 2023?
- Plan: `employment by age; TIME_PERIOD=2023, age=30, geo=FR, sex=F; op=none`
- Selected: `urt_lfe3emp`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/urt_lfe3emp/1.0?c%5Bgeo%5D=FR&c%5BTIME_PERIOD%5D=2023&c%5Bage%5D=Y30&c%5Bsex%5D=F&a

**A04 — completed_exact** (`supported`)
- Question: How many 40-year-old men lived in Italy in 2022?
- Plan: `population by age and sex; TIME_PERIOD=2022, age=40, geo=IT, sex=male; op=none`
- Selected: `demo_pjan`; value: `362241.0`; stopped at: `—`.
- Error: —

**A05 — completed_exact** (`supported`)
- Question: What was the population of 25-year-old women in Spain in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=25, geo=ES, sex=F; op=none`
- Selected: `demo_pjan`; value: `248701.0`; stopped at: `—`.
- Error: —

**A06 — completed_exact** (`supported`)
- Question: What was the population of 35-year-old men in the Netherlands in 2023?
- Plan: `population by age and sex; TIME_PERIOD=2023, age=35, geo=NL, sex=male; op=none`
- Selected: `demo_pjan`; value: `115907.0`; stopped at: `—`.
- Error: —

**B01 — completed_exact** (`supported`)
- Question: How many females aged 20 lived in Belgium in 2024?
- Plan: `female population by age; TIME_PERIOD=2024, age=20, geo=BE, sex=female; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**B02 — completed_exact** (`supported`)
- Question: What was Belgium's female population at age 20 in 2024?
- Plan: `female population by age; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**B03 — unexpected_failure** (`supported`)
- Question: How many Belgian women were 20 years old in 2024?
- Plan: `employment by age; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `urt_lfe3emp`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/urt_lfe3emp/1.0?c%5Bgeo%5D=BE&c%5Bsex%5D=F&c%5Bage%5D=Y20&c%5BTIME_PERIOD%5D=2024&a

**B04 — completed_exact** (`supported`)
- Question: Give the 2024 population of 20-year-old females in Belgium.
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**B05 — unexpected_failure** (`supported`)
- Question: For Belgium in 2024, how many women were age 20?
- Plan: `employment by age and sex; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `ilc_atsb01a`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/ilc_atsb01a/1.0?c%5Bgeo%5D=BE&c%5BTIME_PERIOD%5D=2024&c%5Bage%5D=Y20&c%5Bsex%5D=F&a

**B06 — unexpected_failure** (`supported`)
- Question: What is the female population aged 20 for Belgium, 2024?
- Plan: `female population by age; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `demo_pjangroup`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjangroup/1.0?c%5Bgeo%5D=BE&c%5BTIME_PERIOD%5D=2024&c%5Bage%5D=Y20&c%5Bsex%5D=

**C01 — completed_exact** (`supported`)
- Question: What is the sum of the population of 20-year-old women in Belgium in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=sum`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**C02 — completed_exact** (`supported`)
- Question: Add up the population observation for women aged 20 in Belgium for 2024.
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=sum`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**C03 — unexpected_failure** (`known_unsupported`)
- Question: What is the difference between the population of 20-year-old women in France and Germany in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=DE, sex=F; op=difference`
- Selected: `demo_pjanbroad`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjanbroad/1.0?c%5Bgeo%5D=DE&c%5Bage%5D=Y20&c%5Bsex%5D=F&c%5BTIME_PERIOD%5D=202

**C04 — expected_architecture_limitation** (`known_unsupported`)
- Question: What is the ratio of the population of 20-year-old women in France to Germany in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=DE, sex=F; op=ratio`
- Selected: `demo_pjan`; value: `—`; stopped at: `computation`.
- Error: A ratio computation requires exactly two observations.

**C05 — unexpected_failure** (`known_unsupported`)
- Question: What was the percentage change in the population of 20-year-old women in Belgium from 2023 to 2024?
- Plan: `population by age and sex; TIME_PERIOD=2023-2024, age=20, geo=BE, sex=F; op=percentage_change`
- Selected: `demo_pjan`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan/1.0?c%5Bgeo%5D=BE&c%5Bage%5D=Y20&c%5Bsex%5D=F&c%5BTIME_PERIOD%5D=2023-202

**C06 — expected_architecture_limitation** (`known_unsupported`)
- Question: What is the difference between the female and male populations aged 20 in Belgium in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=BE, sex=F, M; op=difference`
- Selected: `demo_pjan`; value: `—`; stopped at: `filter_resolution`.
- Error: No code matches 'F, M' in codelist 'SEX'.

**D01 — deterministic_validation_failure** (`supported`)
- Question: How many women aged 20 lived in Belgium in 2024?
- Plan: `women aged 20; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `tesem060`; value: `—`; stopped at: `filter_resolution`.
- Error: Unknown dimension 'sex' for data structure 'TESEM060'.

**D02 — deterministic_validation_failure** (`supported`)
- Question: How many women aged 20 lived in Belgium for 2024?
- Plan: `women aged 20; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `tesem060`; value: `—`; stopped at: `filter_resolution`.
- Error: Unknown dimension 'sex' for data structure 'TESEM060'.

**D03 — unexpected_failure** (`supported`)
- Question: How many women aged 20 lived in Belgium on 1 January 2024?
- Plan: `women aged 20; TIME_PERIOD=2024-01-01, age=20, geo=BE, sex=F; op=none`
- Selected: `tesem010`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/tesem010/1.0?c%5Bgeo%5D=BE&c%5BTIME_PERIOD%5D=2024-01-01&c%5Bage%5D=Y20&c%5Bsex%5D=

**D04 — deterministic_validation_failure** (`supported`)
- Question: How many women aged 20 lived in Belgium at the beginning of 2024?
- Plan: `women aged 20; TIME_PERIOD=2024-01-01, age=20, geo=BE, sex=F; op=none`
- Selected: `tesem060`; value: `—`; stopped at: `filter_resolution`.
- Error: Unknown dimension 'sex' for data structure 'TESEM060'.

**D05 — completed_exact** (`supported`)
- Question: What was Belgium's 2024 population of women aged 20?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=BE, sex=F; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**D06 — unexpected_failure** (`supported`)
- Question: How many 20-year-old women lived in Belgium as of January 1, 2024?
- Plan: `employment by age; TIME_PERIOD=2024-01-01, age=20, geo=BE, sex=F; op=none`
- Selected: `urt_lfe3emp`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/urt_lfe3emp/1.0?c%5Bgeo%5D=BE&c%5BTIME_PERIOD%5D=2024-01-01&c%5Bage%5D=Y20&c%5Bsex%

**E01 — model_completion_failure** (`known_unsupported`)
- Question: Give the population of 20-year-old women in Belgium for 2023 and 2024.
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner response has invalid field types.

**E02 — completed_inexact** (`known_unsupported`)
- Question: Give the population of 20-year-old women in France and Germany in 2024.
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=DE, sex=F; op=sum`
- Selected: `demo_pjan`; value: `393310.0`; stopped at: `—`.
- Error: —

**E03 — completed_inexact** (`known_unsupported`)
- Question: Give the male and female populations aged 20 in Belgium in 2024.
- Plan: `population by age; TIME_PERIOD=2024, age=20, geo=BE, sex=male; op=none`
- Selected: `demo_pjan`; value: `69404.0`; stopped at: `—`.
- Error: —

**E04 — expected_architecture_limitation** (`known_unsupported`)
- Question: Give the population aged 20 and 21 in Belgium in 2024.
- Plan: `population by age; TIME_PERIOD=2024, age=20,21, geo=BE; op=none`
- Selected: `demo_poppcctz`; value: `—`; stopped at: `filter_resolution`.
- Error: No code matches '20,21' in codelist 'AGE'.

**E05 — completed_inexact** (`known_unsupported`)
- Question: Give the population of 20-year-old women in Belgium and the Netherlands in 2024.
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=NL, sex=F; op=none`
- Selected: `demo_pjan`; value: `115270.0`; stopped at: `—`.
- Error: —

**E06 — model_completion_failure** (`known_unsupported`)
- Question: Give the population of 20-year-old women in Belgium for 2022, 2023, and 2024.
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner response has invalid field types.

**F01 — safe_failure** (`safe_failure`)
- Question: What was the population of Atlantis in 2024?
- Plan: `population; TIME_PERIOD=2024, geo=ATLANTIS; op=none`
- Selected: `tps00001`; value: `—`; stopped at: `filter_resolution`.
- Error: No code matches 'ATLANTIS' in codelist 'GEO'.

**F02 — safe_failure** (`safe_failure`)
- Question: What was Belgium's population by eye colour in 2024?
- Plan: `population by eye colour; TIME_PERIOD=2024, freq=AN, geo=BE, unit=person; op=none`
- Selected: `—`; value: `—`; stopped at: `dataset_selection`.
- Error: No dataset candidates were found.

**F03 — safe_failure** (`safe_failure`)
- Question: What is the square root of Belgium's population in 2024?
- Plan: `population; TIME_PERIOD=2024, geo=BE, unit=people; op=none`
- Selected: `tps00001`; value: `—`; stopped at: `filter_resolution`.
- Error: Unknown dimension 'unit' for data structure 'TPS00001'.

**F04 — unexpected_failure** (`safe_failure`)
- Question: What was the rate in Belgium in 2024?
- Plan: `rate; TIME_PERIOD=2024, geo=BE; op=none`
- Selected: `ei_lmhr_m`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/ei_lmhr_m/1.0?c%5Bgeo%5D=BE&c%5BTIME_PERIOD%5D=2024&attributes=none&measures=all&co

**F05 — safe_failure** (`safe_failure`)
- Question: How many unicorns were unemployed in Germany in 2024?
- Plan: `unemployment by type of activity; TIME_PERIOD=2024, age=L, geo=DE, sex=M; op=none`
- Selected: `ilc_hch05b`; value: `—`; stopped at: `filter_resolution`.
- Error: No code matches 'L' in codelist 'AGE'.

**F06 — safe_failure** (`safe_failure`)
- Question: What was the happiness score of dragons in Belgium in 2024?
- Plan: `happiness; TIME_PERIOD=2024, age=None, freq=None, geo=BE, sex=None, unit=None; op=none`
- Selected: `—`; value: `—`; stopped at: `dataset_selection`.
- Error: No dataset candidates were found.

### qwen35-constrained

**A01 — completed_exact** (`supported`)
- Question: What was the population of 20-year-old women in Belgium in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**A02 — completed_exact** (`supported`)
- Question: What was the population of 20-year-old men in Germany in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Germany, sex=male; op=none`
- Selected: `demo_pjan`; value: `419725.0`; stopped at: `—`.
- Error: —

**A03 — model_completion_failure** (`supported`)
- Question: How many 30-year-old women lived in France in 2023?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**A04 — completed_exact** (`supported`)
- Question: How many 40-year-old men lived in Italy in 2022?
- Plan: `population by age and sex; TIME_PERIOD=2022, age=40, geo=Italy, sex=male; op=none`
- Selected: `demo_pjan`; value: `362241.0`; stopped at: `—`.
- Error: —

**A05 — completed_exact** (`supported`)
- Question: What was the population of 25-year-old women in Spain in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=25, geo=Spain, sex=female; op=none`
- Selected: `demo_pjan`; value: `248701.0`; stopped at: `—`.
- Error: —

**A06 — completed_exact** (`supported`)
- Question: What was the population of 35-year-old men in the Netherlands in 2023?
- Plan: `population by age and sex; TIME_PERIOD=2023, age=35, geo=Netherlands, sex=male; op=none`
- Selected: `demo_pjan`; value: `115907.0`; stopped at: `—`.
- Error: —

**B01 — completed_exact** (`supported`)
- Question: How many females aged 20 lived in Belgium in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**B02 — completed_exact** (`supported`)
- Question: What was Belgium's female population at age 20 in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**B03 — completed_exact** (`supported`)
- Question: How many Belgian women were 20 years old in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**B04 — completed_exact** (`supported`)
- Question: Give the 2024 population of 20-year-old females in Belgium.
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=Female; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**B05 — completed_exact** (`supported`)
- Question: For Belgium in 2024, how many women were age 20?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**B06 — unexpected_failure** (`supported`)
- Question: What is the female population aged 20 for Belgium, 2024?
- Plan: `population by sex and age; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `cens_hnmga`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/cens_hnmga/1.0?c%5Bgeo%5D=BE&c%5BTIME_PERIOD%5D=2024&c%5Bsex%5D=F&c%5Bage%5D=Y20&at

**C01 — completed_exact** (`supported`)
- Question: What is the sum of the population of 20-year-old women in Belgium in 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=sum`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**C02 — completed_exact** (`supported`)
- Question: Add up the population observation for women aged 20 in Belgium for 2024.
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=sum`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**C03 — model_completion_failure** (`known_unsupported`)
- Question: What is the difference between the population of 20-year-old women in France and Germany in 2024?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner response has invalid field types.

**C04 — model_completion_failure** (`known_unsupported`)
- Question: What is the ratio of the population of 20-year-old women in France to Germany in 2024?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**C05 — model_completion_failure** (`known_unsupported`)
- Question: What was the percentage change in the population of 20-year-old women in Belgium from 2023 to 2024?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner response has invalid field types.

**C06 — model_completion_failure** (`known_unsupported`)
- Question: What is the difference between the female and male populations aged 20 in Belgium in 2024?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**D01 — unexpected_failure** (`supported`)
- Question: How many women aged 20 lived in Belgium in 2024?
- Plan: `population by sex and age; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `cens_hnmga`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/cens_hnmga/1.0?c%5Bgeo%5D=BE&c%5BTIME_PERIOD%5D=2024&c%5Bsex%5D=F&c%5Bage%5D=Y20&at

**D02 — deterministic_validation_failure** (`supported`)
- Question: How many women aged 20 lived in Belgium for 2024?
- Plan: `population by sex and age; TIME_PERIOD=2024, age=20, geo=Belgium, sex=women; op=none`
- Selected: `cens_hnmga`; value: `—`; stopped at: `filter_resolution`.
- Error: No code matches 'women' in codelist 'SEX'.

**D03 — deterministic_validation_failure** (`supported`)
- Question: How many women aged 20 lived in Belgium on 1 January 2024?
- Plan: `population by sex and age; TIME_PERIOD=2024, age=20, geo=Belgium, sex=women; op=none`
- Selected: `cens_hnmga`; value: `—`; stopped at: `filter_resolution`.
- Error: No code matches 'women' in codelist 'SEX'.

**D04 — completed_exact** (`supported`)
- Question: How many women aged 20 lived in Belgium at the beginning of 2024?
- Plan: `population by age and sex; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `demo_pjan`; value: `65231.0`; stopped at: `—`.
- Error: —

**D05 — unexpected_failure** (`supported`)
- Question: What was Belgium's 2024 population of women aged 20?
- Plan: `population by sex and age; TIME_PERIOD=2024, age=20, geo=Belgium, sex=female; op=none`
- Selected: `cens_hnmga`; value: `—`; stopped at: `retrieval`.
- Error: Eurostat request failed: Client error '400 Bad Request' for url 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/cens_hnmga/1.0?c%5Bgeo%5D=BE&c%5BTIME_PERIOD%5D=2024&c%5Bsex%5D=F&c%5Bage%5D=Y20&at

**D06 — model_completion_failure** (`supported`)
- Question: How many 20-year-old women lived in Belgium as of January 1, 2024?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**E01 — model_completion_failure** (`known_unsupported`)
- Question: Give the population of 20-year-old women in Belgium for 2023 and 2024.
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner response has invalid field types.

**E02 — model_completion_failure** (`known_unsupported`)
- Question: Give the population of 20-year-old women in France and Germany in 2024.
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**E03 — model_completion_failure** (`known_unsupported`)
- Question: Give the male and female populations aged 20 in Belgium in 2024.
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**E04 — model_completion_failure** (`known_unsupported`)
- Question: Give the population aged 20 and 21 in Belgium in 2024.
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**E05 — model_completion_failure** (`known_unsupported`)
- Question: Give the population of 20-year-old women in Belgium and the Netherlands in 2024.
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner response has invalid field types.

**E06 — model_completion_failure** (`known_unsupported`)
- Question: Give the population of 20-year-old women in Belgium for 2022, 2023, and 2024.
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner response has invalid field types.

**F01 — safe_failure** (`safe_failure`)
- Question: What was the population of Atlantis in 2024?
- Plan: `population; TIME_PERIOD=2024, geo=Atlantis; op=none`
- Selected: `tps00001`; value: `—`; stopped at: `filter_resolution`.
- Error: No code matches 'Atlantis' in codelist 'GEO'.

**F02 — safe_failure** (`safe_failure`)
- Question: What was Belgium's population by eye colour in 2024?
- Plan: `population by eye colour; TIME_PERIOD=2024, geo=Belgium; op=none`
- Selected: `—`; value: `—`; stopped at: `dataset_selection`.
- Error: No dataset candidates were found.

**F03 — safe_failure** (`safe_failure`)
- Question: What is the square root of Belgium's population in 2024?
- Plan: `population; TIME_PERIOD=2024, geo=Belgium; op=none`
- Selected: `tps00001`; value: `—`; stopped at: `computation`.
- Error: Observation does not contain a unit dimension.

**F04 — safe_failure** (`safe_failure`)
- Question: What was the rate in Belgium in 2024?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**F05 — safe_failure** (`safe_failure`)
- Question: How many unicorns were unemployed in Germany in 2024?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.

**F06 — safe_failure** (`safe_failure`)
- Question: What was the happiness score of dragons in Belgium in 2024?
- Plan: `—`
- Selected: `—`; value: `—`; stopped at: `planner_model_call`.
- Error: Planner returned invalid JSON.
