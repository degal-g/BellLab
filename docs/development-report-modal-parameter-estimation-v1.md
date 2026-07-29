# Development Report: Modal Parameter Estimation v1

## 1. Data

2026-07-29.

## 2. Branch

`feature/modal-parameter-estimation`.

## 3. Estado inicial

Antes de alterar arquivos, a branch foi confirmada com
`git branch --show-current` e a árvore estava limpa. A suíte inicial tinha
865 testes aprovados em `pytest` e 865 testes aprovados em `pytest -W error`.
Nenhuma alteração foi feita diretamente na `main`.

## 4. Arquivos criados

- `belllab/modal_parameters.py`
- `tests/test_modal_parameters.py`
- `docs/development-report-modal-parameter-estimation-v1.md`

## 5. Arquivos alterados

- `belllab/__init__.py`
- `README.md`
- `docs/RFC-0001-scientific-specification.md`

## 6. Princípio científico

A camada implementa sínteses operacionais de valores já presentes em
`ModalHypothesis`. Ela preserva explicitamente:

```text
hipótese modal != modo físico comprovado
frequência representativa != frequência modal exata
tempo de decaimento estimado != constante física invariável
variação entre condições != prova de não linearidade
incerteza operacional != intervalo de confiança físico completo
```

Nenhuma estimativa é promovida a `ModalMode`.

## 7. Estados da estimativa

`ModalParameterEstimateStatus` define estados mutuamente exclusivos:
`valid`, `valid_with_reservations`, `partial`, `insufficient_evidence` e
`invalid_input`.

## 8. Razões

`ModalParameterEstimateReason` separa razões favoráveis, ressalvas,
insuficiências e invalidades. Exemplos favoráveis: frequência suficiente, tau
suficiente, cobertura suficiente e hipótese aceita. Exemplos de ressalva:
hipótese com reservas, match ambíguo, match perto do limiar, contexto de split
ou merge e candidato rejeitado presente. Exemplos de insuficiência: poucos
valores de frequência ou tau, dispersão excessiva e evidência insuficiente.
Exemplos de invalidez: hipótese inválida, frequência inválida, tau inválido e
peso inválido.

## 9. Configuração

`ModalParameterEstimationSettings` controla política de hipótese, frequência,
tau, ressalvas e parâmetros numéricos. Os defaults são conservadores: hipóteses
aceitas e aceitas com ressalvas são permitidas; inconclusivas, rejeitadas e
insuficientes não geram estimativa válida por padrão; frequência exige ao menos
dois valores; tau é opcional, mas quando existe precisa passar limites de
dispersão; bootstrap usa seed determinística `0`.

Enums ou literais validados são usados para métodos. Thresholds são finitos e
não negativos quando presentes; valores `None` desabilitam critérios opcionais.
Fração e nível de confiança são validados no intervalo correto.

## 10. Métodos de localização

`ParameterLocationMethod` suporta média aritmética, mediana, média ponderada,
mediana ponderada, média geométrica e mediana geométrica. A média geométrica é
aplicada apenas a valores positivos. Para tau, os resumos logarítmicos são
sempre preservados.

## 11. Pesos

`ParameterWeightingMethod` suporta pesos uniformes, cobertura de tracking,
qualidade do ajuste de frequência, qualidade do ajuste de amplitude, inverso do
custo de associação e combinação explícita de cobertura, qualidade de
frequência, qualidade de amplitude e custo inverso. Pesos inválidos são
diagnosticados; o vetor público preserva `None` nesses pontos para manter o
contrato finito e não negativo. Pesos normalizados só são emitidos quando todos
os pesos necessários são válidos e a soma é positiva.

## 12. Frequência representativa

`ModalFrequencyEstimate` reúne valores em Hz, labels de condição, IDs de
candidato e track, gravações, pesos, frequência representativa, mínimo, máximo,
range, range relativo, média, mediana, desvio padrão populacional, MAD,
coeficiente de variação, contagens, razões e diagnósticos.

Exemplo usado nos testes:

```text
pp=100.0 Hz, p=100.4 Hz, mf=100.8 Hz, f=101.1 Hz, ff=101.6 Hz
média=100.78 Hz
mediana=100.8 Hz
min=100.0 Hz
max=101.6 Hz
range=1.6 Hz
range relativo=0.01587616590593366
desvio padrão=0.552810998443407 Hz
MAD=0.4 Hz
coeficiente de variação=0.005485324453695247
```

## 13. Trajetória de frequência

`ModalFrequencyTrajectoryEstimate` resume a trajetória na ordem das condições
sem exigir monotonicidade. Para o exemplo acima:

```text
mudanças por passo=(0.4, 0.4, 0.3, 0.5) Hz
mudança total assinada=1.6 Hz
mudança absoluta total=1.6 Hz
slope ordinal=0.39 Hz por passo
intercepto=100.0 Hz
RMSE=0.03741657386773804 Hz
```

O ajuste linear é apenas descritivo no índice ordinal das condições.

## 14. Drift entre condições

O drift é expresso como mudanças assinadas e relativas entre condições
adjacentes, contagens de passos para cima, para baixo, preservados e
indeterminados, além de mudança total assinada e absoluta. Não é classificado
como hardening, softening, causalidade ou prova de não linearidade.

## 15. Incerteza de frequência

`ModalFrequencyUncertainty` suporta desvio padrão amostral, erro padrão, MAD
escalado, bootstrap percentil determinístico e método conservador que combina
dispersão e incertezas individuais disponíveis. Os campos preservam método,
incerteza padrão, limites, nível de confiança, tamanho da amostra, contagem de
bootstrap, seed, incertezas individuais, componentes de dispersão e medição,
validade, razões e diagnósticos.

Bootstrap usa `random.Random(seed)` local, sem alterar estado global.

## 16. Tau representativo

`ModalDecayEstimate` usa `amplitude_tau_s` já existente nos candidatos. Valores
precisam ser positivos e finitos; ausência permanece explícita. Para:

```text
tau=(4.0, 4.2, 3.9, 4.1) s
média aritmética=4.05 s
mediana=4.05 s
média geométrica=4.048456119233489 s
mediana geométrica=4.049691346263318 s
log média=1.398335603313769
log mediana=1.3986406674150764
log desvio padrão=0.027618972568965783
log MAD=0.02439508208471608
log range=0.07410797215372211
```

## 17. Taxa de decaimento

`ModalDecayRateEstimate` deriva apenas relações matemáticas documentadas a
partir de tau válido. A convenção é:

```text
A(t) = A0 exp(-t / tau)
amplitude_decay_rate = 1 / tau
```

Quando energia é proporcional a `A^2`, a taxa energética documentada é
`2 / tau`.

## 18. Tempos em dB

Com `tau = 2.0 s`:

```text
taxa de amplitude=0.5 1/s
taxa de energia=1.0 1/s
tempo até 1/e=2.0 s
tempo até -20 dB=4.605170185988092 s
tempo até -40 dB=9.210340371976184 s
tempo até -60 dB=13.815510557964275 s
```

Os tempos seguem a convenção de decaimento de amplitude, não uma inferência
modal física.

## 19. Incerteza de tau

`ModalDecayUncertainty` opera no domínio logarítmico. Métodos suportados:
desvio padrão logarítmico, erro padrão logarítmico, MAD logarítmico escalado e
bootstrap percentil no domínio log. Os limites são transformados de volta para
segundos e podem ser assimétricos.

## 20. Proveniência

`ModalParameterProvenance` registra `hypothesis_id`, `source_chain_id`,
`candidate_ids`, `match_ids`, labels de condição, gravações, contagens de fonte
de frequência e tau, matches ambíguos ou perto do limiar, contextos de split e
merge, fingerprint determinístico da configuração e diagnósticos. O fingerprint
não usa timestamp.

## 21. Política de status

A precedência implementada é:

1. entrada inválida;
2. hipótese não permitida pela configuração;
3. frequência insuficiente;
4. tau insuficiente quando exigido;
5. violação crítica de dispersão;
6. estimativa parcial;
7. estimativa válida com ressalvas;
8. estimativa válida.

Hipóteses rejeitadas ou com evidência insuficiente podem ser auditadas, mas não
viram estimativas válidas silenciosamente.

## 22. Determinismo

IDs de estimativa e fingerprints são hashes determinísticos de hipótese, dados,
proveniência e configuração. A ordem de entrada das hipóteses não altera a
ordem final, os IDs ou as contagens. Bootstrap é determinístico para a mesma
seed.

## 23. Imutabilidade

As APIs não alteram hipóteses, cadeias, candidatos ou matches. A implementação
usa tuplas nas saídas, não reordena listas de entrada in-place, não usa caches
globais mutáveis e não altera estado global de aleatoriedade.

## 24. Testes

Foram adicionados 81 testes em `tests/test_modal_parameters.py`, cobrindo:
frequência básica, trajetória monotônica e não monotônica, dispersão e limites
inclusivos, métodos de localização, métodos de peso, pesos inválidos, tau,
taxa de decaimento, tempos em dB, incertezas de frequência e tau, hipóteses com
status distintos, resultado global, determinismo, perturbação local,
imutabilidade e invariantes numéricos.

## 25. Resultado final

A suíte cresceu de 865 para 946 testes. A validação de integração com `pytest`
aprovou 946 testes após a implementação. A validação final da rodada executou
`pytest`, `pytest -W error`, `python3 -m compileall -q belllab tests`,
`git diff --check` e `git status`; os comandos de teste e integridade passaram,
e `git status` mostrou somente os arquivos modificados/criados da rodada antes
do commit.

## 26. Limitações

Esta camada não implementa `ModalMode`, fator Q, largura de banda, frequência
física definitiva, ajuste de oscilador amortecido completo, hardening,
softening, prova de linearidade, prova de não linearidade, split, merge,
fechamento de lacunas, associação não adjacente, matching global, troca de
energia, acoplamento modal, causalidade, machine learning, visualizações
finais, pipeline completo de experimento, leitura de áudio ou exportação final
de relatório.

## 27. Próximos passos

Próximas rodadas podem adicionar exportação tabular auditável, integração com
relatórios científicos e visualizações diagnósticas. Fator Q, largura de banda
ou modelos físicos só devem ser tratados em camadas separadas e com premissas
científicas próprias.
