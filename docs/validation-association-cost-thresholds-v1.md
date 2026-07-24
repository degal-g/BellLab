# Validation Report — Association Cost Thresholds v1

**Data:** 2026-07-23  
**Estado inicial:** 154 testes aprovados.

## Escopo e arquivos

Esta rodada alterou `belllab/config.py`, `tests/test_tracking.py`, `README.md` e
este relatório. Ela validou exclusivamente `maximum_association_cost` e
`near_threshold_ratio`; não adicionou métricas, algoritmos, cruzamentos,
lacunas, candidatos ou modos modais.

## Custo máximo de associação

`maximum_association_cost` é o gate aplicado ao custo total selecionado:

`selected_cost = frequency_cost_component + amplitude_cost_component`

A política é inclusiva: `selected_cost <= maximum_association_cost`. Com
tolerância de 10 Hz e peso frequencial 2, os casos reais controlados produziram
custos 0,4 (abaixo de 0,5, aceito), 0,5 (igual a 0,5, aceito) e 0,6 (acima de
0,5, rejeitado). Na rejeição, não houve diagnóstico de associação aceita; a
trajetória original foi retomada após uma lacuna de um quadro e o pico rejeitado
iniciou a trajetória de ID 1.

Não existe constante implícita igual a 1. Um custo 1,6 foi aceito com máximo
2,0, enquanto um custo 0,6 foi rejeitado com máximo 0,5.

Os testes com pesos maiores que 1 separaram os componentes. Em amplitude linear,
a distância frequencial 2 Hz gerou componente 0,4 e a distância de amplitude
0,5 gerou componente 1,0, totalizando 1,4. Em dBFS, a mudança de -10 para -4
dBFS gerou distância 0,3 e componente 0,6; somada ao componente frequencial
0,4, produziu custo total 1,0. Um caso adicional confirmou que o gate
frequencial admitiu 2 Hz, mas o custo total 1,4 foi rejeitado pelo máximo 1,2.

Valores zero, negativos, `NaN`, `+inf` e `-inf` para
`maximum_association_cost` são rejeitados com a exigência de valor finito e
estritamente positivo.

## Proximidade do limiar

Para associações aceitas, a definição validada é inclusiva:

`near_threshold = selected_cost >= near_threshold_ratio * maximum_association_cost`

Com máximo 2,0 e razão 0,5, o limiar foi 1,0. Custos reais 0,8, 1,0 e 1,2
produziram, respectivamente, `False`, `True` e `True`. A decisão usa o custo
total configurado, sem referência implícita ao custo unitário.

Com razão 0, toda associação aceita, inclusive custo zero, foi marcada como
próxima. Com razão 1, custo 1,8 abaixo do máximo 2,0 não foi marcado, enquanto
custo exatamente 2,0 foi marcado.

Razões negativas, maiores que 1, `NaN`, `+inf` e `-inf` são rejeitadas. A
configuração exige valor finito no intervalo fechado `[0, 1]`. A validação
comportamental revelou e corrigiu apenas a aceitação indevida de `NaN`.

## Independência e reprodutibilidade

Os diagnósticos foram separados em matching real:

- cenário A: custo 0,1, margem operacional 0,05, `ambiguous=True` e
  `near_threshold=False`;
- cenário B: custo 1,8, sem alternativa e sem margem operacional,
  `ambiguous=False` e `near_threshold=True`.

Logo, ambiguidade mede proximidade entre alternativas, enquanto
`near_threshold` mede proximidade absoluta do custo ao gate. Os casos no máximo,
no limiar, acima do limiar e os dois cenários de independência foram repetidos
sem aleatoriedade; associações, custos, diagnósticos e IDs permaneceram iguais.

Os testes de associações aceitas verificam explicitamente quadro, ID da
trajetória, índice do pico, custo selecionado, distância e unidade frequencial,
componentes de frequência e amplitude, distância de amplitude, margem,
ambiguidade, proximidade e a soma exata dos componentes com `pytest.approx`.

## Resultado

Foram adicionados **29 casos de teste**, elevando a suíte de **154 para 183
testes**.

- `pytest`: 183 aprovados;
- `pytest -W error`: 183 aprovados;
- `python3 -m compileall -q belllab tests`: aprovado;
- `git diff --check`: aprovado;
- não há ferramenta estática adicional configurada em `pyproject.toml`.

Limitações: os custos são diagnósticos operacionais, não probabilidades; a
margem não prova identidade física; o gate frequencial anterior permanece
separado do gate de custo total. Não foram implementados `ModalCandidate`,
`ModalMode`, fator Q, análise modal, GUI ou processamento em lote.
