# Relatorio de desenvolvimento: cadeias operacionais de candidatos entre condicoes

## 1. Data

2026-07-27.

## 2. Branch

`feature/cross-condition-candidate-chains`.

## 3. Estado inicial

Antes de qualquer alteracao nesta rodada:

- `git branch --show-current`: `feature/cross-condition-candidate-chains`;
- `pytest`: 772 testes coletados, 772 aprovados;
- `pytest -W error`: 772 testes coletados, 772 aprovados.

## 4. Arquivos alterados

- `belllab/candidate_chains.py` criado;
- `belllab/__init__.py` atualizado;
- `tests/test_cross_condition_candidate_chains.py` criado;
- `README.md` atualizado;
- `docs/RFC-0001-scientific-specification.md` atualizado;
- `docs/development-report-cross-condition-candidate-chains-v1.md` criado.

## 5. Principio cientifico

A implementacao preserva explicitamente:

```text
cadeia operacional de candidatos
!= modo fisico comprovado
!= familia modal comprovada
!= identidade modal persistente
!= prova de linearidade
!= prova de nao linearidade
```

Uma cadeia significa apenas uma sequencia consistente de correspondencias
operacionais ja aceitas entre condicoes dinamicas adjacentes. Ela nao retroage
para transformar matches locais em identidade fisica comprovada.

## 6. Sequencia nominal

A ordem canonica vem de `DYNAMIC_LABEL_ORDER`:

```text
pp -> p -> mf -> f -> ff
```

A API aceita apenas subsequencias contiguas, como `pp -> p`, `p -> mf -> f`,
`mf -> f -> ff` e a sequencia completa. Sequencias invertidas, repetidas, com
saltos, vazias, de uma unica condicao ou com rotulos desconhecidos sao rejeitadas.

## 7. Contratos publicos

Foram criados:

- `AdjacentAssociationSequence`;
- `CandidateChainNode`;
- `CrossConditionCandidateChain`;
- `CrossConditionCandidateChainResult`.

Foram exportados em `belllab/__init__.py` junto com as APIs publicas novas.

## 8. Construcao das cadeias

`build_cross_condition_candidate_chains(...)` recebe resultados ja calculados por
associacao adjacente. Ela valida a sequencia, coleta as referencias candidatas,
usa somente `matches` adjacentes aceitos como arestas dirigidas e constroi
componentes lineares maximais. Nenhum custo e recalculado e nenhuma aresta nova
e criada.

Matches audit-only com candidatos rejeitados (`match.accepted=False`) nao sao
usados como arestas; os candidatos permanecem preservados em cadeias unitarias.

## 9. Particao completa

O resultado garante que cada candidato da sequencia aparece em exatamente uma
cadeia. Cada match adjacente aceito aparece em exatamente uma cadeia. Candidatos
sem match aparecem como cadeias unitarias. A soma de todos os nos das cadeias
iguala `candidate_count`.

No teste completo, 14 candidatos foram particionados em 5 cadeias.

## 10. Cadeias completas

Exemplo quantitativo da suite:

```text
100.0 -> 101.0 -> 102.0 -> 103.0 -> 104.0 Hz
A -> D -> G -> J -> L
```

Essa cadeia atravessa `pp -> p -> mf -> f -> ff`, possui 4 matches e e completa
para a sequencia solicitada.

## 11. Cadeias parciais

Exemplos do teste completo:

```text
B -> E -> H
F -> I -> K -> M
```

A primeira termina em `mf` e registra desaparecimento quando sustentado por
`mf -> f`. A segunda comeca em `p` e registra emergencia quando sustentada por
`pp -> p`.

## 12. Cadeias unitarias

Exemplos do teste completo:

```text
C
N
```

Ambas possuem `match_count=0`, `condition_count=1` e agregados de custo ausentes
como `None`, nao zero.

## 13. Emergencia

Uma cadeia que comeca depois da primeira condicao solicitada pode carregar
`starts_as_emerging=True` somente quando o resultado adjacente anterior ja
registrou o candidato em `EmergingCandidate`.

Exemplo: `F -> I -> K -> M` comeca em `p` na sequencia completa e herda a
emergencia do par `pp -> p`. A mesma cadeia e completa e nao emergente quando a
sequencia solicitada e apenas `p -> mf -> f -> ff`.

## 14. Desaparecimento

Uma cadeia que termina antes da ultima condicao solicitada pode carregar
`ends_as_disappearing=True` somente quando o resultado adjacente seguinte ja
registrou o candidato em `DisappearingCandidate`.

Exemplo: `B -> E -> H` termina em `mf` na sequencia completa e herda
desaparecimento do par `mf -> f`. Para a subsequencia `pp -> p -> mf`, ela e
completa e termina na fronteira da sequencia.

## 15. Trajetoria de frequencia

Cada cadeia preserva:

- `frequency_trajectory_hz`;
- mudancas assinadas por passo;
- mudancas relativas assinadas por passo com denominador simetrico;
- classificacoes operacionais de deslocamento;
- frequencia inicial e final;
- mudanca total absoluta assinada;
- mudanca total relativa simetrica;
- contagens de passos para cima, para baixo, preservados e indeterminados.

No exemplo `100.0 -> 101.0 -> 102.0 -> 103.0 -> 104.0 Hz`, as mudancas por
passo sao `(1.0, 1.0, 1.0, 1.0) Hz` e a mudanca relativa total e `4.0 / 102.0`.

## 16. Custos

Os custos locais dos matches sao preservados sem agregacao opaca. No exemplo
completo, a cadeia `A -> D -> G -> J -> L` tem custos:

```text
(0.5, 0.5, 0.5, 0.5)
```

Logo, minimo, maximo e media sao `0.5`. Cadeias sem matches mantem minimo,
maximo, media, maior custo normalizado e menor margem como `None`.

## 17. Ambiguidade

Uma cadeia com match ambiguo continua valida, mas expoe:

- `contains_ambiguous_match`;
- `ambiguous_match_ids`;
- `ambiguous_match_positions`;
- `ambiguous_assignment_margins`.

No teste `pp:100.0` contra `p:99.9,100.1`, o empate gera margem `0.0`; o match
selecionado permanece na cadeia linear e a alternativa aparece como cadeia
unitaria emergente.

## 18. Near-threshold

Near-threshold e preservado por match local. No teste:

```text
100.0 -> 100.2 -> 102.0 -> 102.2 Hz
```

os custos sao `(0.1, 0.9, 0.1)` com `maximum_association_cost=1.0` e
`near_threshold_ratio=0.8`. O match intermediario e identificado na posicao 1.

## 19. Contexto de split

Possiveis splits ja diagnosticados por associacao adjacente sao anexados quando
um no da cadeia participa do contexto. No teste `p:200.0` contra
`mf:198.5,201.5`, os custos das alternativas sao `(0.75, 0.75)`. A cadeia
principal permanece linear e o candidato alternativo nao entra na mesma cadeia.

## 20. Contexto de merge

Possiveis merges ja diagnosticados por associacao adjacente sao anexados sem
fundir cadeias. No teste `mf:298.5,301.5` contra `f:300.0`, os custos sao
`(0.75, 0.75)`. O candidato nao selecionado permanece em outra cadeia.

## 21. Determinismo

Os IDs das cadeias sao hashes deterministas baseados na sequencia solicitada,
nas identidades canonicas dos candidatos e nos IDs dos matches adjacentes. Nao
ha UUID aleatorio, timestamp, endereco de memoria ou contador global mutavel.

Os testes confirmam igualdade normalizada com pares em ordem canonica, invertida
e com listas locais de matches, emergentes e desaparecidos embaralhadas.

## 22. Imutabilidade

As entradas nao sao modificadas. Os resultados adjacentes continuam com suas
tuplas originais e construcao repetida produz o mesmo resultado normalizado.

Um teste de perturbacao local altera apenas a frequencia do candidato `D`; os
IDs das cadeias permanecem estaveis e somente a trajetoria/custos da cadeia
`A -> D -> G -> J -> L` mudam.

## 23. Testes adicionados

Foi criado `tests/test_cross_condition_candidate_chains.py` com 28 testes
coletados, cobrindo:

- sequencia completa;
- cadeias completas, parciais e unitarias;
- emergencia e desaparecimento;
- lacunas sem fechamento por frequencia;
- ambiguidade;
- near-threshold;
- contexto de split e merge;
- subsequencias;
- entradas invalidas;
- determinismo;
- perturbacao local;
- imutabilidade;
- invariantes numericos;
- matches audit-only rejeitados.

## 24. Resultado final

Apos a implementacao, a suite passou de 772 para 800 testes.

## 25. Limitacoes

Esta rodada nao implementa:

- `ModalMode`;
- familia modal fisica;
- associacao direta `pp -> ff`;
- fechamento de lacunas;
- matching global alternativo;
- otimizacao de rota global;
- resolucao de split;
- resolucao de merge;
- inferencia de nao linearidade;
- classificacao de hardening ou softening;
- analise causal;
- machine learning;
- visualizacoes finais;
- leitura de audio;
- recomputacao de FFT/STFT;
- recomputacao de candidatos;
- recomputacao de associacoes locais.

## 26. Proximos passos

- Avaliar relatorios tabulares de auditoria das cadeias sem visualizacao final;
- definir criterios futuros para promover, se cientificamente justificavel,
  cadeias operacionais a hipoteses modais sem confundir contrato operacional com
  identidade fisica;
- estudar, em rodada separada, como representar familias modais sem reutilizar
  split/merge diagnostico como conclusao fisica.
