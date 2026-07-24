# Development Report — Track Characterization v1.2

**Data:** 2026-07-23  
**Estado inicial:** 100 testes aprovados.

Esta revisão completou a validação de margens bidirecionais e contratos mínimos
da caracterização sem criar `ModalCandidate` ou `ModalMode`.

Para cada associação Húngara, a margem de linha é a diferença entre o custo
selecionado e a segunda alternativa válida da mesma trajetória; a margem de
coluna aplica a mesma regra às trajetórias concorrentes pelo mesmo pico. A
margem operacional é a menor margem disponível, ou a única margem disponível.
Ausência de alternativa é `None`, nunca infinito público. Margens abaixo de
`ambiguity_margin` são diagnósticos heurísticos, não probabilidades.

`SpectralTrackCharacterization` agora valida ID, unidade, limites de frequência,
cobertura, tau e compatibilidade entre método e unidade. A unidade de amplitude
permanece explícita. Para linear, o ajuste usa `ln(A)` e `tau=-1/m`; para dBFS,
usa dB/s e `tau=-20/(m ln(10))`, somente com inclinação negativa.

Foram acrescentados 5 testes, elevando a suíte para **105 passed**. Eles
confirmam tau linear conhecido, tau dBFS de 1 s para
`-8.685889638 dB/s`, Hz, distância relativa simétrica e cents: oitava = 1200
cents e semitom = 100 cents. A distância relativa usa
`abs(f2-f1)/max(abs(f1),abs(f2))` para frequências positivas; zero–zero é zero
e uma frequência não positiva isolada é inadmissível (`inf`). Cents não admite
frequências não positivas.

`pytest`, `compileall` e `git diff --check` passaram. Permanecem pendentes uma
camada pública detalhada de diagnósticos por associação, testes mais amplos de
`amplitude_weight` e validação com gravações reais. Nenhuma trajetória foi
interpretada como modo físico.
