# Relatório de desenvolvimento: evidência operacional de redistribuição modal de energia v1

Data: 2026-07-29

Branch: `feature/modal-energy-exchange`

Estado inicial: branch confirmada fora da `main`, árvore limpa, `pytest` e
`pytest -W error` com 1015 testes aprovados antes das alterações.

## Arquivos criados

- `belllab/modal_energy_exchange.py`
- `tests/test_modal_energy_exchange.py`
- `docs/development-report-modal-energy-exchange-v1.md`

## Arquivos alterados

- `README.md`
- `docs/RFC-0001-scientific-specification.md`
- `belllab/__init__.py`

## Princípio científico

A camada implementada identifica somente evidência operacional compatível com
possível redistribuição aparente de energia entre componentes. Ela preserva:

```text
anticorrelação entre envelopes != transferência física de energia comprovada
crescimento tardio != excitação interna comprovada
atraso temporal != causalidade
batimento aparente != acoplamento modal comprovado
conservação aproximada de soma de energias != sistema fechado
evidência operacional de troca != prova de troca física
```

Os termos "possível", "aparente", "operacional" e "compatível com" fazem parte
da interpretação científica da camada.

## Status

`ModalEnergyExchangeStatus` define estados mutuamente exclusivos:

- `supported`
- `supported_with_reservations`
- `inconclusive`
- `not_supported`
- `insufficient_evidence`
- `invalid_input`

## Razões

`ModalEnergyExchangeReason` separa evidências favoráveis, ressalvas,
inconclusões, ausência de suporte, insuficiências e invalidades. As razões
incluem tendências opostas, correlação negativa significativa, correlação com
lag, crescimento tardio, recuperação, dominância alternada, proxy de energia do
par aproximadamente conservado, batimento possível, tracking ambíguo,
near-threshold, sobreposição temporal insuficiente, eixo temporal incompatível,
envelope ausente e valores inválidos.

## Configuração

`ModalEnergyExchangeSettings` torna explícitas as políticas de entrada, janela
temporal, representação de amplitude, normalização, suavização, tendências,
correlação, significância, proxy de energia, batimento e decisão. Defaults são
conservadores: gates de tendências opostas e correlação negativa são exigidos,
seeds são determinísticas e valores opcionais `None` desabilitam critérios.

As invariantes verificam frações em `[0,1]`, tempos não negativos,
frequências positivas quando aplicáveis, contagens positivas, seed inteira ou
`None`, pesos não negativos e ausência de thresholds ocultos.

## Representação de amplitude

`ModalAmplitudeRepresentation` cobre:

- amplitude linear;
- amplitude normalizada;
- amplitude em dB;
- potência relativa;
- energia operacional.

Entradas de amplitude linear rejeitam valores negativos. Amplitudes dB são
convertidas para amplitude linear apenas para construir o proxy operacional.
Zero não é usado como substituto de ausência.

## Proxy de energia

`ModalEnergyProxy` implementa `E_proxy(t) proportional A(t)^2`. O resultado é
chamado de proxy de energia, energia operacional ou energia relativa aparente,
nunca energia física. A ponderação por frequência existe apenas por política
explícita e fica desabilitada por padrão.

## Preparação de envelopes

`ModalEnvelopeSeries` preserva `source_id`, `hypothesis_id`, `candidate_id`,
`track_id`, `recording_id`, `dynamic_label`, tempos, amplitudes, amplitudes
normalizadas, proxy de energia, máscara válida, contagem, intervalo temporal,
amostragem, normalização, suavização, razões e diagnósticos.

A fonte canônica reutilizada é `Envelope.times_s`/`Envelope.amplitudes` ou
`SpectralTrack.times_s`/`SpectralTrack.amplitudes`. A função
`prepare_modal_envelope_series` também aceita tuplas explícitas já calculadas.

## Alinhamento temporal

`ModalEnvelopeAlignment` trabalha na faixa comum de duas séries. As políticas
são eixos idênticos, interseção, interpolação linear e downsampling. Não há
extrapolação. Em um caso de teste com eixos deslocados em 0,05 s e passo de
reamostragem 0,1 s, a interpolação começa no primeiro tempo realmente
amostrado por ambos, evitando cruzar a faixa disponível.

## Tendências

`ModalEnvelopeTrendEvidence` calcula inclinações descritivas por regressão
linear, diferença por segmentos, mediana das derivadas ou início-fim. Em um
exemplo sintético, `A(t)` decresce de 1,0 para 0,4 em 1 s enquanto `B(t)` cresce
de 0,0 para aproximadamente 0,9165; as tendências são opostas. Isso não é
interpretado como lei física.

## Crescimento tardio

`ModalDelayedGrowthEvidence` exige crescimento posterior ao início da janela,
fração mínima configurável e duração mínima opcional. Um crescimento de 0,20
para 0,23 com limite de 15% é aceito no limiar; 0,20 para 0,229 é rejeitado.
Um pico isolado de ruído é rejeitado quando a duração mínima configurada não é
atendida.

## Recuperação

`ModalAmplitudeRecoveryEvidence` distingue decaimento inicial, mínimo local e
recuperação posterior. No teste `1,0 -> 0,4 -> 0,7`, a recuperação relativa é
0,5 do decaimento inicial e é marcada como suporte operacional quando o limiar
é 25%.

## Correlação e lag

`ModalEnvelopeCorrelationEvidence` implementa Pearson e Spearman. A convenção é:

```text
lag > 0
```

significa que mudanças no componente A precedem mudanças no componente B no
eixo alinhado. A convenção é descritiva e não causal. Em um caso sintético com
atraso de 0,2 s, o melhor lag negativo é recuperado como aproximadamente
0,2 s.

## Significância

A significância operacional usa deslocamento circular ou permutação em blocos
com `random_seed` explícito. O RNG global não é alterado. Em testes com seed
fixa, execuções repetidas produzem o mesmo p-valor.

## Energia do par

`ModalPairEnergyEvidence` calcula `energy_a`, `energy_b`, `pair_energy`, média,
desvio padrão, range relativo, coeficiente de variação e fração estável. No
exemplo `A(t)^2 + B(t)^2 = 1`, o range relativo é 0 e a evidência auxiliar é
marcada como aproximadamente conservada. Isso não declara sistema fechado.

## Alternância de dominância

`ModalAlternatingDominanceEvidence` usa razão mínima com histerese. Em um caso
com energias alternando `A>B`, `B>A`, `A>B`, duas trocas são detectadas. Pequenas
oscilações perto da igualdade são rejeitadas pela histerese.

## Possível batimento

`ModalBeatingEvidence` registra compatibilidade entre separação frequencial e
período de modulação. Para 100 Hz e 101 Hz, `T_beat = 1 / |100 - 101| = 1 s`.
Um envelope modulado com período observado de aproximadamente 1 s é marcado
como possível batimento e gera ressalva, não evidência adicional de troca.

## Score

`ModalEnergyExchangeScore` combina componentes auditáveis: tendências opostas,
correlação negativa, correlação com lag, crescimento tardio, recuperação,
dominância alternada, proxy de energia do par, qualidade de tracking, penalidade
de batimento e penalidade por evidência ausente. O score é limitado a `[0,1]` e
gates obrigatórios prevalecem.

No exemplo sintético com `A(t)^2 + B(t)^2 = 1`, o score normalizado é 0,81 com
os pesos padrão e status `supported`.

## Política de status

A precedência implementada é:

1. entrada inválida;
2. origem não permitida;
3. envelope ausente;
4. sobreposição insuficiente;
5. amostras insuficientes;
6. gate obrigatório falho;
7. evidência insuficiente;
8. evidências contraditórias;
9. ausência de suporte;
10. suporte com ressalvas;
11. suporte.

Uma origem `ModalParameterEstimate` inválida não gera evidência suportada; ela
retorna `insufficient_evidence` com `unsupported_source_status`.

## Determinismo

IDs são determinísticos e baseados nas séries, parâmetros e fingerprint da
configuração. Pares são canonicalizados por `source_id`, impedindo duplicação
A-B/B-A em análises globais. Ordenações invertidas ou embaralhadas produzem o
mesmo resultado normalizado. Mudanças locais alteram apenas pares que usam a
fonte perturbada.

## Imutabilidade

As entradas não são modificadas. Preparação, interpolação, suavização e
permutação operam sobre cópias. O bootstrap/permutação usa `random.Random` local
e não altera o estado global de RNG. Não há cache global mutável.

## Testes

Foram adicionados 60 testes em `tests/test_modal_energy_exchange.py`, cobrindo:

- troca aparente sintética;
- ausência de troca;
- possível batimento;
- correlação com lag;
- crescimento tardio;
- recuperação;
- proxy de energia do par;
- dominância alternada;
- eixos temporais;
- amplitudes inválidas;
- correlação Pearson/Spearman;
- significância determinística;
- status global;
- determinismo;
- perturbação local;
- imutabilidade;
- invariantes de configuração e contrato.

## Resultado final

Após a implementação, a suíte cresce de 1015 para 1075 testes. A validação final
executada nesta rodada é documentada no commit associado.

## Limitações

Esta rodada não implementa prova de transferência física de energia,
causalidade, modelo de osciladores acoplados, coeficiente de acoplamento,
ajuste físico completo, separação de picos sobrepostos, resolução de split,
resolução de merge, fechamento de lacunas, associação não adjacente, hardening,
softening, prova de linearidade, prova de não linearidade, machine learning,
visualizações finais, pipeline completo de experimento, leitura de áudio ou
exportação final de relatório científico.

## Próximos passos

- Integrar seleção automática de fontes de envelope a partir de resultados de
  tracking quando houver um objeto agregador de experimento.
- Adicionar relatórios tabulares de evidência sem promover conclusões físicas.
- Comparar políticas de significância em séries autocorrelacionadas reais.
