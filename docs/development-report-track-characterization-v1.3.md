# Development Report — Track Characterization v1.3

**Data:** 2026-07-23. **Estado inicial:** 105 testes aprovados.

Esta revisão expõe `TrackAssignmentDiagnostic` em cada associação aceita, sem
publicar matrizes de custo. Ele registra custo, distância e componentes de
frequência/amplitude, margens por linha e coluna, margem operacional,
ambiguidade e proximidade do limite. A margem operacional é o mínimo das
margens disponíveis; `None` significa que não há segunda alternativa válida.
Margens são diagnósticos heurísticos, não probabilidades.

`SpectralTrackingSettings` passou a declarar `maximum_association_cost=2.0` e
`near_threshold_ratio=0.9`. O gate de tolerância frequencial permanece
independente; custos totais acima do máximo são rejeitados, e custos a partir
de `ratio * maximum` são próximos do limite. O custo é auditável como soma das
componentes ponderadas de frequência e amplitude.

Foi adicionado `TrackAmplitudeFit`, contrato imutável para futura integração
estruturada de ajustes. Ele valida unidade, contagens, sucesso/falha e tau. A
caracterização existente continua compatível. Nenhum candidato modal foi criado.

Um novo teste verifica a decomposição pública do custo; a suíte final possui
**106 passed**. `pytest`, `compileall` e `git diff --check` passaram.

Limitações: a caracterização ainda precisa migrar integralmente para
`TrackAmplitudeFit`, e cenários extensos de amplitude_weight e cruzamentos com
gravações reais permanecem a próxima validação. Tracking não é análise modal.
