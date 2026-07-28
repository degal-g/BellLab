# Relatorio de desenvolvimento: camada de hipotese modal operacional v1

## 1. Data

2026-07-28.

## 2. Branch

`feature/modal-hypothesis-layer`.

## 3. Estado inicial

Antes de qualquer alteracao nesta rodada:

- `git branch --show-current`: `feature/modal-hypothesis-layer`;
- arvore de trabalho limpa;
- `pytest`: 800 testes coletados, 800 aprovados;
- `pytest -W error`: 800 testes coletados, 800 aprovados.

Nenhuma alteracao direta foi feita na `main`.

## 4. Arquivos criados e alterados

Criados:

- `belllab/modal_hypotheses.py`;
- `tests/test_modal_hypotheses.py`;
- `docs/development-report-modal-hypothesis-layer-v1.md`.

Alterados:

- `belllab/__init__.py`;
- `README.md`;
- `docs/RFC-0001-scientific-specification.md`.

## 5. Principio cientifico

A camada preserva explicitamente:

```text
cadeia operacional de candidatos
!= hipotese modal aceita

hipotese modal aceita
!= modo fisico comprovado

persistencia entre condicoes
!= identidade fisica definitiva

deslocamento de frequencia
!= prova de nao linearidade

contexto de split ou merge
!= divisao ou fusao fisica
```

Uma `ModalHypothesis` significa somente que uma cadeia operacional satisfez os
criterios ativos para ser tratada como hipotese modal operacional ao longo da
sequencia solicitada.

## 6. Diferenca entre cadeia e hipotese

`CrossConditionCandidateChain` continua sendo uma sequencia operacional de
candidatos conectados por matches adjacentes ja aceitos. A nova camada nao
recalcula associacoes, nao cria arestas novas e nao muda a cadeia.

`ModalHypothesis` e uma decisao auditavel derivada dessa cadeia. A hipotese
mantem o `source_chain_id`, a propria cadeia, evidencias separadas, score,
status e razoes. Cadeias rejeitadas, inconclusivas ou com evidencia
insuficiente continuam presentes no resultado.

## 7. Estados

Foram criados estados publicos em `ModalHypothesisStatus`:

- `accepted`;
- `accepted_with_reservations`;
- `inconclusive`;
- `rejected`;
- `insufficient_evidence`;
- `invalid_input`.

Os estados sao mutuamente exclusivos. O campo `accepted` so e verdadeiro em
`accepted` e `accepted_with_reservations`.

## 8. Razoes

`ModalHypothesisReason` preserva razoes tipadas para suporte, ressalva,
rejeicao e ausencia de evidencia. Entre elas:

- `sufficient_cross_condition_persistence`;
- `sufficient_frequency_continuity`;
- `sufficient_tracking_quality`;
- `sufficient_decay_consistency`;
- `sufficient_impact_evidence`;
- `complete_chain`;
- `partial_but_supported_chain`;
- `singleton_chain`;
- `too_few_conditions`;
- `too_few_matches`;
- `frequency_discontinuity`;
- `excessive_frequency_variation`;
- `excessive_association_cost`;
- `excessive_ambiguity`;
- `excessive_near_threshold_fraction`;
- `insufficient_tracking_quality`;
- `inconsistent_decay`;
- `missing_required_decay`;
- `missing_required_impact_evidence`;
- `possible_split_context`;
- `possible_merge_context`;
- `rejected_candidate_present`;
- `invalid_chain`;
- `insufficient_evidence`.

## 9. Configuracao

`ModalHypothesisSettings` cobre criterios de cobertura, frequencia,
associacao, tracking, decaimento, pre-impacto, split/merge e decisao. Valores
`None` desabilitam limites opcionais. Pesos sao nao negativos, fracoes ficam em
`[0, 1]`, limites sao finitos e custos configurados nao podem ser infinitos.

Defaults conservadores:

- cadeia completa requerida;
- minimo de 2 condicoes e 1 match;
- frequencia dominante no score (`frequency_continuity_weight=4.0`);
- limites de frequencia, custo, tracking e tau habilitados;
- decaimento e impacto nao obrigatorios por padrao;
- split/merge geram ressalva por padrao, nao rejeicao;
- ausencia opcional de tracking, tau e pre-impacto vira ressalva por padrao.

## 10. Cobertura

`ModalHypothesisCoverageEvidence` mede cobertura relativa a sequencia
solicitada, nao apenas ao trecho atravessado pela cadeia.

Exemplo real dos testes:

```text
cadeia: p -> mf -> f
sequencia solicitada: pp -> p -> mf -> f -> ff
observed_condition_count = 3
requested_condition_count = 5
condition_coverage_fraction = 3 / 5 = 0.6
complete_across_requested_sequence = False
partial = True
```

Com `require_complete_chain=False`, `allow_partial_chains=True` e cobertura
minima `0.6`, o resultado foi `accepted_with_reservations`. Com a configuracao
conservadora, a mesma cadeia foi `rejected`.

## 11. Continuidade frequencial

`ModalHypothesisFrequencyEvidence` preserva frequencias, mudancas assinadas,
mudancas absolutas, mudancas relativas simetricas, RMSE da trajetoria e
contagens de passos operacionais ja herdadas da cadeia.

Exemplo aceito:

```text
100.0, 100.2, 100.1, 100.3, 100.2 Hz
signed_step_changes_hz = (0.2, -0.1, 0.2, -0.1)
total_absolute_change_hz = 0.6
trajectory_rmse_from_mean_hz = 0.10198039027185603
maximum_step_change_hz = 0.2
```

Tambem passaram trajetorias crescente, decrescente e nao monotonica continua.
A trajetoria `100.0, 100.2, 150.0, 150.2, 150.4 Hz` foi rejeitada por
`frequency_discontinuity`. Nenhum teste classifica hardening, softening,
linearidade ou nao linearidade.

## 12. Qualidade de associacao

`ModalHypothesisAssociationEvidence` preserva IDs dos matches, custos por match,
media, minimo, maximo, contagem/fração de ambiguidade e near-threshold, alem de
margens quando disponiveis na cadeia.

Exemplo aceito da cadeia forte:

```text
match_costs = (0.02, 0.01, 0.02, 0.01)
mean_match_cost = 0.015
maximum_match_cost = 0.02
```

Os testes cobrem custo baixo, custo alto, media aceitavel com maximo
inaceitavel, limite inclusivo, ambiguidade, near-threshold, margem ausente e
margem abaixo do minimo.

## 13. Tracking

`ModalHypothesisTrackingEvidence` usa somente os campos de `CandidateReference`:
cobertura, fracoes ambiguas, fracoes near-threshold, margens e RMSE de ajuste de
frequencia. Nao recalcula tracking.

Os testes verificam cobertura alta, cobertura baixa, alta ambiguidade, alta
fracao near-threshold, RMSE alto, dados completos, dados ausentes parcialmente e
todos os dados ausentes. A politica `missing_tracking_evidence_policy` controla
se a ausencia e aceita, vira ressalva, gera `insufficient_evidence` ou rejeita.

## 14. Decaimento

`ModalHypothesisDecayEvidence` usa apenas `amplitude_tau_s` existente em cada
referencia candidata. Aceita somente `tau > 0`, preserva ausencias e avalia
consistencia em dominio logaritmico.

Exemplos reais:

```text
4.0, 4.2, 3.9, 4.1, 4.0  -> passa
4.0, 4.8, 3.7, 4.5, 4.2  -> passa
1.0, 8.0, 0.7, 10.0, 2.0 -> inconsistent_decay
```

Tau `0.0`, negativo e infinito foram ignorados como valores invalidos e
contados como ausentes; nao foram convertidos para zero. Esta rodada nao calcula
fator Q nem largura de banda.

## 15. Pre-impacto

`ModalHypothesisImpactEvidence` reutiliza `impact_excited` e as classificacoes
existentes (`impact_emergent`, `impact_amplified`,
`persistent_background_tone`, `reexcited_preexisting_component`, etc.).

Exemplos dos testes:

```text
5/5 candidatos sustentados por impacto -> accepted
3/5 sustentados, minimo 0.6          -> accepted
1/5 sustentado, minimo 0.6           -> rejected
0/5 evidencias disponiveis e requisito habilitado -> insufficient_evidence
```

A evidencia pre-impacto permanece operacional e nao estabelece causalidade.

## 16. Contexto de split e merge

`ModalHypothesisStructuralContext` preserva `possible_split_contexts` e
`possible_merge_contexts` vindos das cadeias. A politica configuravel permite
rejeitar, aceitar com ressalvas ou preservar apenas como diagnostico.

Os testes confirmam:

- nenhum branching de hipotese;
- nenhuma fusao de hipoteses;
- ligacao a uma unica cadeia;
- nenhuma conclusao de split ou merge fisico.

## 17. Score

`ModalHypothesisScore` expoe componentes:

- cobertura;
- frequencia;
- associacao;
- tracking;
- decaimento;
- impacto;
- penalidade estrutural;
- penalidade por evidencia ausente.

Cada componente tem valor, peso, valor ponderado e disponibilidade em
`ModalHypothesisScoreComponent`. O score normalizado fica em `[0, 1]` e a
frequencia tem peso dominante por padrao. Componentes ausentes nao entram no
denominador; a ausencia e registrada por politica e penalidade, nao por zero
implicito.

Na cadeia forte, o score normalizado foi maior que `0.9`. O teste de soma
recalcula:

```text
raw_score = sum(weighted_value) / sum(weights) - penalties
normalized_score = raw_score
```

## 18. Precedencia de decisao

A decisao segue:

1. entrada invalida;
2. gate obrigatorio;
3. evidencia insuficiente;
4. rejeicao por criterios configurados;
5. aceitacao com ressalvas;
6. aceitacao;
7. inconclusivo.

O score nao substitui gates. Exemplo real:

```text
require_decay_evidence=True
nenhum tau disponivel
score passa o limiar configurado de 0.1
status = insufficient_evidence
```

## 19. Hipoteses aceitas

Uma cadeia completa `pp -> p -> mf -> f -> ff` com frequencias proximas,
custos baixos, tracking estavel, tau consistente, evidencia de impacto e sem
split/merge foi `accepted`, com:

```text
accepted = True
requires_review = False
```

## 20. Hipoteses aceitas com ressalvas

Cadeias parciais permitidas, matches ambiguos dentro da politica, matches
near-threshold permitidos e contextos de possivel split/merge em modo reserva
produzem `accepted_with_reservations` quando o score passa o limiar de reserva.

## 21. Hipoteses inconclusivas

Uma cadeia forte com `minimum_acceptance_score` acima do score observado e sem
criterio critico falho foi classificada como `inconclusive`.

## 22. Hipoteses rejeitadas

Foram testadas rejeicoes por:

- descontinuidade frequencial;
- variacao frequencial excessiva;
- custo de associacao excessivo;
- ambiguidade excessiva;
- near-threshold excessivo;
- qualidade de tracking insuficiente;
- decaimento inconsistente;
- impacto insuficiente quando requerido;
- split/merge configurados para rejeicao;
- cadeia parcial quando cadeia completa e obrigatoria.

## 23. Evidencia insuficiente

Cadeias unitarias com defaults conservadores e cadeias sem tau quando
`require_decay_evidence=True` geram `insufficient_evidence`. Entradas invalidas
geram `invalid_input`, sem promover objeto externo a cadeia.

## 24. Determinismo

IDs de hipotese sao hashes deterministas baseados na sequencia solicitada, no
`source_chain_id`, nas identidades canonicas dos candidatos e nos IDs dos
matches adjacentes. Nao usam UUID, timestamp ou contador global.

Os testes com cadeias em ordem original, invertida e embaralhada produziram a
mesma forma normalizada de IDs, status, scores, componentes, razoes,
diagnosticos e contagens. Mudancas apenas na ordem dos diagnosticos da cadeia
nao alteraram o resultado normalizado.

## 25. Imutabilidade

As entradas permanecem imutadas. Testes com `deepcopy` confirmam que cadeias,
candidatos e matches nao sao modificados, que diagnosticos nao sao reordenados
in-place e que builds repetidos produzem resultados identicos.

Uma perturbacao local de frequencia alterou apenas a hipotese correspondente,
mantendo o ID da hipotese e da cadeia estaveis; a outra hipotese permaneceu
identica.

## 26. Testes adicionados

Foi criado `tests/test_modal_hypotheses.py` com 65 testes coletados, cobrindo:

- cadeia completa forte;
- cadeia parcial;
- cadeia unitaria;
- trajetorias frequenciais estavel, crescente, decrescente, nao monotonica e
  descontinua;
- limites inclusivos;
- custos, ambiguidade, near-threshold e margens;
- tracking completo, ruim e ausente;
- tau consistente, moderado, inconsistente, ausente e invalido;
- evidencia pre-impacto;
- split e merge;
- todos os status;
- score e pesos;
- conjunto de cadeias;
- determinismo;
- perturbacao local;
- imutabilidade;
- invariantes de configuracao e valores numericos.

## 27. Resultado final

A suite passou de 800 para 865 testes.

Validacao final da rodada:

```text
pytest
865 passed

pytest -W error
865 passed

python3 -m compileall -q belllab tests
OK

git diff --check
OK
```

## 28. Limitacoes

Esta rodada nao implementa:

- `ModalMode`;
- frequencia modal fisica definitiva;
- fator Q;
- largura de banda;
- ajuste fisico de oscilador;
- hardening;
- softening;
- prova de linearidade;
- prova de nao linearidade;
- resolucao de split;
- resolucao de merge;
- fechamento de lacunas;
- associacao nao adjacente;
- matching global;
- troca de energia;
- causalidade;
- machine learning;
- visualizacoes finais;
- pipeline completo de experimento;
- leitura de audio;
- exportacao final de relatorio cientifico.

## 29. Proximos passos

- Refinar limiares de `ModalHypothesisSettings` apenas com validacao
  experimental documentada;
- projetar tabelas de auditoria para hipoteses sem transformar a camada em
  relatorio cientifico final;
- estudar, em rodada separada, contratos para modos fisicos comprovados sem
  retroagir essa interpretacao para candidatos, matches, cadeias ou hipoteses
  operacionais.
