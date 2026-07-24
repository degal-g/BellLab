# Validation Report — Track Characterization v1.4

**Data:** 2026-07-23. **Estado inicial:** 106 testes aprovados.  
**Resultado final:** 110 testes aprovados.

Esta rodada validou os contratos públicos de auditabilidade. Foram acrescentados
testes para diagnóstico de associação com margens disponíveis ou ausentes e
para rejeição de `NaN` em custos; `TrackAmplitudeFit` agora rejeita contagens
incoerentes e tau contraditório. A suíte existente já valida recuperação de
tau linear (0.2, 1 e 3 s), tau dBFS de 1 s para -8.685889638 dB/s, semitom de
100 cents e oitava de 1200 cents.

O custo máximo é inclusivo (`selected_cost <= maximum_association_cost`) e o
limiar próximo é `selected_cost >= near_threshold_ratio * maximum_association_cost`.
Margens, custos e tau seguem sendo diagnósticos operacionais; não são
probabilidades nem interpretação modal. `pytest`, `compileall` e
`git diff --check` passaram. Não há ferramenta estática configurada.

Limitações: a próxima rodada deve ampliar cenários completos de
`amplitude_weight`, cruzamentos e não finitos em sinais reais. Nenhum
`ModalCandidate` ou `ModalMode` foi criado.
