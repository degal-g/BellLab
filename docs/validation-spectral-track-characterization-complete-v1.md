# Validation Report — Complete Spectral Track Characterization v1

**Data:** 2026-07-23  
**Estado inicial:** 183 testes aprovados.

## Escopo e arquivos alterados

Esta rodada completou somente a descrição matemática e estatística de
trajetórias espectrais. Foram alterados `belllab/types.py`,
`belllab/tracking.py`, `belllab/__init__.py`, `tests/test_tracking.py`,
`tests/test_track_characterization_complete.py`, `README.md`,
`docs/RFC-0001-scientific-specification.md` e este relatório.

Uma trajetória espectral continua sendo uma sequência matemática de picos. A
caracterização não promove trajetórias a candidatos ou modos físicos e não
calcula fator Q, classificação, agrupamento ou energia modal.

## Frequência canônica e origem

Cada observação usa a frequência interpolada quando ela está presente, é finita
e estritamente positiva. Caso contrário, usa a frequência do bin. A origem
global é `interpolated` quando todas vieram da interpolação, `bin` quando todas
vieram do bin e `mixed` quando ambas as origens participaram. O caso misto
também recebe `mixed_frequency_source`.

Somente pares finitos de tempo e frequência participam das métricas. A
caracterização registra contagens disponíveis, finitas e descartadas. Séries
sem valor finito mantêm métricas opcionais em `None`, nunca NaN, e recebem
diagnósticos de descarte e insuficiência.

## Métricas e regressão de frequência

Foram adicionados valores inicial e final, média, mediana, mínimo, máximo,
desvio padrão, deriva total, pico a pico e estabilidade relativa. A deriva é
`final - initial`; o pico a pico é `maximum - minimum`. A estabilidade relativa
é `standard_deviation / median` apenas para mediana estritamente positiva.
Valores menores indicam menor dispersão relativa, não confiança estatística ou
validade modal.

`TrackFrequencyFit` é a fonte canônica da regressão
`linear_frequency_drift`. Ele contém sucesso, método, inclinação em Hz/s,
intercepto em Hz, R² opcional, RMSE em Hz, contagens, intervalo, motivo de falha
e diagnósticos. Os aliases antigos de inclinação e RMSE encaminham somente para
esse objeto.

Resultados reais dos testes:

- 100, 102 e 104 Hz em 0, 1 e 2 s: inclinação 2 Hz/s, intercepto 100 Hz,
  RMSE aproximadamente zero e R² 1;
- 104, 102 e 100 Hz: inclinação -2 Hz/s e deriva -4 Hz;
- 100, 102 e 101 Hz: média e mediana 101 Hz, desvio padrão
  0,8164965809 Hz, inclinação 0,5 Hz/s, intercepto 100,5 Hz, RMSE
  0,7071067812 Hz e R² 0,25;
- frequência constante 100 Hz: ajuste bem-sucedido, inclinação
  aproximadamente zero, RMSE aproximadamente zero e `r_squared=None`.

Um ponto falha com `insufficient_frequency_points`; tempos todos iguais falham
com `non_distinct_frequency_times`. Mediana zero produz
`relative_frequency_stability=None` e `zero_frequency_median`.

## Amplitude

As métricas agora incluem valores inicial e final, média, mediana, mínimo,
máximo, desvio padrão, pico a pico, inclinação na escala original, contagens e
proporções de incremento, decremento e constância. A tolerância absoluta para
uma diferença constante é `1e-12`. Quando há diferenças válidas, as três
frações ficam em `[0, 1]` e somam 1.

Os casos 3, 2, 1 e 1, 2, 3 recuperaram inclinações -1 e +1 por segundo e
frações de decremento e incremento iguais a 1. A série constante 2, 2, 2
produziu fração constante 1. A série oscilante 1, 2, 1, 2 produziu incremento
2/3 e decremento 1/3. Em dBFS, -3, -6 e -9 geraram média -6 dBFS e inclinação
-3 dB/s; essa média permanece no domínio de nível, sem conversão para amplitude
linear.

`TrackAmplitudeFit` continua sendo a fonte canônica do ajuste de amplitude.
Valores não finitos são excluídos tanto das descrições quanto do ajuste. Com
1, NaN, +inf, 0,5 e 0,25, três pontos foram finitos e dois descartados; a média
finita foi 0,5833333333. Com menos de duas amplitudes finitas, inclinação e
proporções permanecem `None`.

## Cobertura e lacunas

A caracterização expõe primeiro e último quadro, tempos inicial e final,
duração observada, observações, extensão total em quadros, cobertura,
quantidade de lacunas, quadros internos ausentes, maior lacuna e alcance do
quadro final da análise.

`coverage_fraction = observation_count / frame_span_count`. Por exemplo,
quadros 0, 2 e 3 cobrem 3/4 do intervalo, com uma lacuna de um quadro e alcance
do quadro final 3. Quadros 0, 2, 5 e 6 cobrem 4/7, com duas lacunas, três
quadros ausentes e maior lacuna de dois quadros. Duração, observações e lacunas
permanecem grandezas distintas.

## Invariantes e reprodutibilidade

`TrackFrequencyFit` rejeita falhas sem motivo, sucesso incompleto, regressão em
estado de falha, valores não finitos, RMSE negativo, contagens incoerentes,
menos de dois pontos usados, intervalo invertido e diagnósticos vazios ou
duplicados.

`SpectralTrackCharacterization` valida limites e coerência de frequência e
amplitude, deriva, pico a pico, estabilidade, origem, contagens, compatibilidade
dos dois ajustes, proporções e soma, unidade, duração, ordem temporal, cobertura
e lacunas. Objetos com métricas NaN ou infinitas são rejeitados.

Caracterizações com frequência mista, amplitude oscilante e lacuna foram
executadas duas vezes. Métricas, ajustes, diagnósticos, contagens e origem
foram idênticos.

## Resultado final

Foram adicionados **51 casos de teste**, elevando a suíte de **183 para 234
testes**.

- `pytest`: 234 aprovados;
- `pytest -W error`: 234 aprovados;
- `python3 -m compileall -q belllab tests`: aprovado;
- `git diff --check`: aprovado;
- não há ferramenta estática adicional configurada em `pyproject.toml`.

Limitações: regressões são descritivas e não probabilísticas; R² constante é
indefinido; médias dBFS são médias de nível; a origem interpolada não é uma
incerteza; cobertura não prova continuidade física. Nenhum candidato modal,
novo modo físico, fator Q ou critério de promoção modal foi implementado.
