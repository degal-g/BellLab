# Development Report: Real Experiment Pipeline v1

## 1. Data

2026-07-29.

## 2. Branch

`feature/real-experiment-pipeline`.

## 3. Estado inicial

Antes de qualquer alteração, a branch foi confirmada como
`feature/real-experiment-pipeline`, diferente de `main`, com árvore de trabalho
limpa. A suíte inicial apresentou `1159 passed` em `pytest` e `1159 passed` em
`pytest -W error`.

## 4. Arquivos criados

- `belllab/experiment_pipeline.py`
- `tests/test_experiment_pipeline.py`
- `examples/analyze_real_experiment.py`
- `docs/development-report-real-experiment-pipeline-v1.md`

## 5. Arquivos alterados

- `README.md`
- `docs/RFC-0001-scientific-specification.md`
- `belllab/__init__.py`

## 6. Princípio científico

O pipeline real é apenas um orquestrador reprodutível de camadas científicas
existentes. Ele preserva explicitamente:

```text
pipeline concluído
≠ análise fisicamente válida por definição

resultado produzido
≠ evidência suficiente

ausência de erro computacional
≠ ausência de problema científico

configuração padrão
≠ configuração universal

comparação entre condições
≠ prova de não linearidade

hipótese modal
≠ modo físico comprovado

evidência de possível redistribuição
≠ transferência física comprovada
```

## 7. Definição do experimento

`ExperimentDefinition` registra `experiment_id`, nome, descrição, espécime,
instrumento, localização, operador, data de aquisição, ordem dinâmica,
gravações, referência opcional, equipamentos, notas, taxa/canais esperados,
configuração, metadados e diagnósticos. Quando o ID não é fornecido, ele é
derivado deterministicamente de conteúdo e metadados declarados; nenhum
timestamp ou UUID é usado.

## 8. Gravações

`ExperimentRecordingDefinition` registra caminho WAV, rótulo dinâmico,
`take_index`, grupo de repetição, canal, offsets, polaridade, microfone, ganho,
notas e metadados. Caminhos vazios, rótulos fora de `pp/p/mf/f/ff`, canal
negativo, distância não positiva e offsets incoerentes são rejeitados.

## 9. Repetições

`ExperimentReplicatePolicy` define:

- `analyze_all_separately`
- `explicit_reference`
- `select_by_quality_after_analysis`
- `combine_summaries_only`
- `reject_multiple_replicates`

Nenhuma política mistura formas de onda por padrão. `ExperimentReplicateQuality`
registra clipping, duração, proxy de SNR, cobertura de tracking, candidatos,
componentes do score, ranking e motivo de seleção.

## 10. Configuração

`ExperimentPipelineSettings` habilita/desabilita cada estágio e compõe as
configurações existentes de análise temporal, FFT, STFT, tracking, candidatos,
pré-impacto, excitação, comparação dinâmica, associação, hipóteses, parâmetros,
Q e energia operacional. Dependências inválidas são rejeitadas antes da
execução.

## 11. Grafo de estágios

`EXPERIMENT_PIPELINE_STAGE_ORDER` e
`EXPERIMENT_PIPELINE_STAGE_DEPENDENCIES` registram o grafo:

```text
load → temporal/global_spectrum/STFT
STFT → tracking
tracking → preimpact/modal_candidates/modal_energy_exchange
modal_candidates → within_condition
within_condition → cross_condition
cross_condition → candidate_chains
candidate_chains → modal_hypotheses
modal_hypotheses → modal_parameters
modal_parameters → modal_q
```

## 12. Carregamento

`load_experiment_recordings(...)` usa somente `belllab.io.load_wav`. O caminho
original é preservado, o hash SHA-256 do arquivo é registrado, o canal é
selecionado explicitamente e offsets são aplicados apenas quando configurados.
Arquivos não são modificados.

## 13. Validação

`ExperimentInputValidation` registra contagens, labels presentes/ausentes,
IDs duplicados, arquivos ausentes, taxas de amostragem, canais, durações,
uniformidade e razões de falha. Diferenças de sample rate ou canais são
permitidas somente quando a configuração permitir.

## 14. Análise por gravação

`ExperimentRecordingAnalysisResult` preserva resultados temporais, espectrais,
STFT, picos por frame, tracking, caracterizações de tracks, pré-impacto,
excitação, caracterização espectral global/tempo-resolvida, candidatos e stage
results. Uma falha posterior não remove resultados anteriores.

## 15. Análise por condição

`ExperimentConditionAnalysisResult` agrupa gravações pela condição dinâmica,
preserva todas as repetições e registra a gravação de referência selecionada
por política explícita. `associate_candidates_within_condition(...)` é
reutilizada quando candidatos existem.

## 16. Comparação entre condições

A ordem canônica `pp → p → mf → f → ff` é reutilizada. O pipeline chama
`compare_dynamic_conditions(...)` para métricas descritivas e
`associate_candidates_across_adjacent_conditions(...)` somente para pares
nominalmente adjacentes presentes. Em `pp, p, f, ff`, o pipeline cria `pp→p` e
`f→ff`, mas não cria `p→f`.

## 17. Chains

`build_cross_condition_candidate_chains(...)` é chamado por trechos contíguos
de associações adjacentes. Lacunas dinâmicas quebram a sequência e geram
resultados separados.

## 18. Hipóteses

`build_modal_hypotheses(...)` é chamado sobre chains existentes. O pipeline não
cria `ModalMode` e não interpreta uma hipótese aceita como identidade modal
física.

## 19. Parâmetros

`estimate_modal_parameters(...)` é reutilizado para frequência representativa,
trajetória, drift, tau, taxa de decaimento e incertezas operacionais. Ausência
continua como `None`.

## 20. Q

`estimate_modal_q_factors(...)` é reutilizado. Quando há espectro associado à
condição de origem, ele é fornecido como fonte operacional de bandwidth; quando
a configuração usa apenas decaimento, `Q = pi f tau` permanece condicionado à
convenção de decaimento de amplitude já documentada.

## 21. Energia operacional

`evaluate_modal_energy_exchange(...)` é chamado somente dentro de cada
gravação, usando tracks existentes. Envelopes de gravações diferentes não são
comparados como se compartilhassem o mesmo eixo físico de tempo.

## 22. Execução parcial

Estágios desabilitados são `skipped`; dependências ausentes são `blocked`;
objetos não calculados permanecem `None`. Casos testados incluem somente load,
load+validação e pipeline completo com energia desabilitada.

## 23. Retomada

`validate_precomputed_experiment_stage(...)` e
`resume_experiment_analysis(...)` implementam reuso conservador em memória. A
retomada valida ID de gravação, fingerprint de arquivo, configuração ou versão
quando esses valores são fornecidos. Não há cache persistente nesta rodada.

## 24. Erros

Foram adicionadas exceções públicas:

- `ExperimentDefinitionError`
- `ExperimentInputError`
- `ExperimentPipelineDependencyError`
- `ExperimentStageExecutionError`
- `ExperimentPrecomputedResultError`

Com continuação habilitada, erros são preservados em resultados estruturados.

## 25. Proveniência

`ExperimentProvenance` registra experimento, gravações, caminhos, hashes de
conteúdo, labels, repetições selecionadas, fingerprint de settings, versão do
BellLab, ordem de estágios, estágios completos/omitidos/falhos e metadados de
entrada.

## 26. Determinismo

IDs de gravação, experimento e análise são baseados em conteúdo, configuração,
fingerprints e versão. Testes cobrem ordem de gravações e labels embaralhadas,
repetição de execução, fingerprints e perturbação local de arquivo.

## 27. Imutabilidade

Contratos são dataclasses congelados; listas de entrada são convertidas para
tuplas; metadados são preservados em `MappingProxyType`. O pipeline não
reordena listas in-place, não modifica arquivos WAV e não usa cache mutável
global.

## 28. Testes

Foram adicionados 45 testes em `tests/test_experiment_pipeline.py`, cobrindo:

- contratos públicos;
- validação de definição e settings;
- fingerprints de arquivo e configuração;
- WAVs temporários;
- seleção explícita de canal;
- offsets;
- sample rates e canais distintos;
- execução parcial;
- pipeline até hipóteses, parâmetros, Q e energia operacional;
- condição ausente sem associação não adjacente;
- múltiplas repetições;
- falha estruturada de arquivo;
- fail-fast;
- resultados pré-calculados;
- determinismo;
- perturbação local;
- imutabilidade.

Exemplo quantitativo usado nos testes: série `pp,p,mf,f,ff` com frequências
`300, 301, 302, 303, 304 Hz`, duração `1.2 s` e segundo componente em
`520, 521, 522, 523, 524 Hz`. O pipeline recupera candidatos, constrói
associações adjacentes, chains, hipóteses, parâmetros e estimativas de Q por
decaimento configurado.

## 29. Resultado final

Antes da rodada: `1159 passed`. Após a implementação, a suíte esperada passa a
`1204` testes quando `pytest` executa a nova cobertura junto aos testes
anteriores. A validação final deve incluir `pytest`, `pytest -W error`,
`python3 -m compileall -q belllab tests examples`, `git diff --check` e
`git status`.

## 30. Limitações

Esta rodada não implementa CLI final, GUI, notebook final, dashboards,
relatório HTML/PDF, banco de dados, cache persistente, processamento
distribuído, serviço web, upload, descoberta automática de arquivos,
calibração automática, reamostragem silenciosa, downmix silencioso, seleção
física definitiva de modo, causalidade, prova de não linearidade, resolução
física de split/merge, associação não adjacente ou fechamento de lacunas.

## 31. Próximos passos

- Adicionar CLI estável sobre `ExperimentDefinition`.
- Definir esquema serializável para salvar resultados intermediários.
- Integrar exportação científica final somente após estabilizar contratos.
- Ampliar políticas de seleção de repetição sem misturar formas de onda.
