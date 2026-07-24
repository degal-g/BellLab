# Development Report — Pre-impact Evidence v1

**Data:** 2026-07-23  
**Estado inicial:** 303 testes aprovados.

## Motivação científica

Esta rodada implementa evidência operacional de que um impacto alterou uma
componente espectral rastreada. Uma linha presente antes do impacto não é
automaticamente ruído irrelevante: ela pode coincidir com uma ressonância e ser
amplificada ou reexcitada. A análise preserva a trajetória, a caracterização e
candidatos aceitos ou rejeitados; não promove resultados a modos físicos.

Foram alterados `belllab/config.py`, `belllab/types.py`,
`belllab/modal_candidates.py`, `belllab/__init__.py`, `README.md` e
`docs/RFC-0001-scientific-specification.md`. Foram adicionados
`belllab/preimpact.py`, `tests/test_preimpact.py` e este relatório.

## Referência temporal e janelas

O instante canônico é `ImpactReport.impact_time_s`. Cada tempo da trajetória é
convertido para `time_s - impact_time_s`. `PreImpactAnalysisSettings` declara
as janelas relativas; os padrões são `[-1,0, -0,1] s` e `[+0,02, +0,30] s`.
O intervalo entre -0,1 e +0,02 s exclui a transição. Esses valores são
configuráveis, finitos e validados; a janela pré termina antes do impacto e a
pós não começa antes dele.

Detecção pré-impacto exige contagem mínima, cobertura temporal mínima,
mediana finita e, opcionalmente, nível mínimo. Cobertura é a fração da duração
da janela abrangida entre a primeira e a última observação finita. Ausência
total é diferente de poucos pontos ou pontos não finitos.

## Estatísticas robustas

A mediana é o nível representativo canônico de cada janela. Média, desvio
padrão e inclinação no domínio original permanecem disponíveis. Valores não
finitos são descartados com contagens e diagnósticos. O teste com níveis pré
1, 1 e 100 produziu média 34, mediana 1 e comparou o pós-impacto contra a
mediana, evitando domínio do outlier.

## Amplitude linear

Para medianas lineares estritamente positivas:

`post_to_pre_ratio = post_median / pre_median`

`impact_level_change_db = 20 * log10(post_to_pre_ratio)`

Zero, valores negativos ou não finitos não entram no logaritmo e não são
substituídos por epsilon. Nesses casos, razão ou diferença ficam `None` e o
diagnóstico `linear_level_change_db_unavailable` é registrado.

Os testes recuperaram aumentos de 12 dB, 6 dB e 5,9 dB. Com limiar 6 dB, 12 e
6 passaram, enquanto 5,9 não passou. A comparação inclusiva usa tolerância
numérica de `1e-12`.

## dBFS

Em dBFS a comparação permanece no domínio de nível:

`impact_level_change_db = post_median_dbfs - pre_median_dbfs`

A razão equivalente `10 ** (difference_db / 20)` é derivada apenas da
diferença; a série completa não é exponenciada. Os mesmos casos 12, 6 e 5,9 dB
confirmaram a política inclusiva.

## Classificações operacionais

O conjunto público é:

- `not_detected_preimpact`;
- `impact_emergent`;
- `impact_amplified`;
- `persistent_background_tone`;
- `preexisting_decay`;
- `reexcited_preexisting_component`;
- `insufficient_preimpact_data`;
- `insufficient_postimpact_data`;
- `indeterminate`.

Uma linha ausente antes e presente depois foi `impact_emergent`, com
`impact_excited=True` e sem razão inválida. A política de considerar ausência
como evidência é configurável.

Uma linha presente e inalterada produziu diferença 0 dB,
`persistent_background_tone`, `impact_excited=False` e
`background_contaminated=True`. Uma linha pós-impacto com metade do nível
linear prévio produziu -6,0205999133 dB e nenhuma falsa excitação.

## Pré-decaimento e reexcitação

Os níveis pré 3, 2 e 1, nos tempos -0,9, -0,5 e -0,1 s, recuperaram inclinação
-2,5 por segundo e `preimpact_decay_detected=True`. Sem novo aumento, a
classificação foi `preexisting_decay`. Com níveis pós 8, 6 e 4, a mediana
subiu de 2 para 6, ou +9,5424250944 dB, produzindo
`reexcited_preexisting_component`. Decaimento pós-impacto pode ser exigido,
mas não é obrigatório por padrão.

## Insuficiência e invariantes

Janela vazia, um ponto, todos os valores NaN/+inf/-inf e insuficiência apenas
no pós foram testados separadamente. Falhas carregam motivo e classificação de
insuficiência correspondentes; nenhuma métrica pública propaga NaN.

`PreImpactEvidence` valida ID, unidade, contagens, finitude, dispersões, razão
positiva, classificações, sucesso/falha, evidência de excitação, presença
prévia para contaminação, inclinações de decaimento e diagnósticos imutáveis,
únicos e não vazios.

## Integração opcional com candidatos

`ModalCandidateSettings` recebeu `require_impact_excitation`,
`reject_persistent_background_tone` e
`minimum_impact_level_increase_db`. Todos permanecem desabilitados por padrão.
Quando habilitados, geram `CandidateCriterionResult` e preservam candidatos
rejeitados e suas razões.

Linhas emergentes e preexistentes amplificadas passam por
`require_impact_excitation`. Um tom persistente só é rejeitado quando o critério
correspondente é explicitamente habilitado. A presença pré-impacto, por si só,
nunca é critério de rejeição.

`analyze_candidates_preimpact` preserva ordem, candidatos aceitos e rejeitados
e não muta entradas. Execuções repetidas produziram evidências, classificações
e contagens idênticas.

## Validação

Foram adicionados **63 testes**, elevando a suíte de **303 para 366 testes**.

- `pytest`: 366 aprovados;
- `pytest -W error`: 366 aprovados;
- `python3 -m compileall -q belllab tests`: aprovado;
- `git diff --check`: aprovado;
- não há ferramenta estática adicional configurada em `pyproject.toml`.

## Limitações

Aumento de nível é evidência operacional, não prova causalidade ou identidade
modal. A cobertura usa somente tempos disponíveis na trajetória e não inventa
piso de ruído. Os limiares e janelas dependem do experimento. Esta versão não
implementa associação entre gravações, fator Q, amortecimento físico,
agrupamento harmônico, energia modal, visualização final, lote ou conversão
para `ModalMode`.
