# Development report — cross-condition candidate association v1

**Data:** 2026-07-27  
**Branch:** `feature/cross-condition-candidate-association`  
**Estado inicial:** 713 testes aprovados com `pytest` e `pytest -W error`.

## 1. Arquivos alterados

- `belllab/cross_condition.py`
- `belllab/within_condition.py`
- `belllab/__init__.py`
- `tests/test_cross_condition_candidate_association.py`
- `README.md`
- `docs/RFC-0001-scientific-specification.md`
- `docs/development-report-cross-condition-candidate-association-v1.md`

## 2. Princípio científico

A nova camada preserva explicitamente:

```text
correspondência operacional entre candidatos
≠ identidade modal física comprovada
≠ modo preservado
≠ prova de linearidade ou não linearidade
```

Uma correspondência entre dois candidatos significa somente que eles são
suficientemente compatíveis, sob a configuração ativa, para serem tratados como
correspondentes operacionais em duas condições adjacentes.

## 3. Pares adjacentes

O contrato público `AdjacentDynamicConditionPair` aceita somente:

```text
pp -> p
p -> mf
mf -> f
f -> ff
```

Rótulos desconhecidos, rótulos iguais, pares invertidos, saltos como
`pp -> mf` e a comparação direta `pp -> ff` são rejeitados pela função de baixo
nível.

## 4. Custo

`CrossConditionCandidateAssociationSettings` mantém frequência como componente
dominante por padrão. Termos opcionais permanecem desativados por peso zero ou
limite `None`.

Componentes auditáveis:

- frequência;
- estabilidade frequencial;
- drift frequencial;
- RMSE do ajuste frequencial;
- tau;
- qualidade do ajuste de amplitude;
- ambiguidade de tracking;
- proximidade do limiar de tracking;
- diferença de margem;
- evidência pré-impacto.

O campo `total_cost` é validado como soma exata de `cost_components`.

## 5. Frequência

Cada par candidato-candidato registra:

```text
absolute = abs(f_high - f_low)
relative = abs(f_high - f_low) / ((f_high + f_low) / 2)
log = abs(log2(f_high / f_low))
```

Somente frequências positivas são aceitas. A normalização do custo frequencial
usa o primeiro limite habilitado em ordem absoluta, relativa e logarítmica; os
demais limites ativos continuam gates inclusivos.

Exemplo real dos testes:

```text
100.0 Hz -> 101.0 Hz
absolute = 1.0 Hz
relative = 1 / 100.5
log = abs(log2(101 / 100))
frequency cost = 1 / 2 = 0.5
```

## 6. Deslocamento de frequência

A classificação operacional de deslocamento é:

```text
frequency_preserved
frequency_shifted_up
frequency_shifted_down
frequency_shift_indeterminate
```

O padrão considera frequência preservada quando a diferença absoluta é menor
ou igual a 0.5 Hz. Essa tolerância é configurável por diferença absoluta ou
relativa. O resultado registra somente sentido de deslocamento; não usa termos
de interpretação física, linearidade ou não linearidade.

Casos reais dos testes:

```text
100.0 -> 100.4 Hz: frequency_preserved
100.0 -> 100.6 Hz: frequency_shifted_up
100.0 -> 99.4 Hz: frequency_shifted_down
100.0 -> 100.5 Hz: frequency_preserved no limite
```

## 7. Tau

Tau usa distância logarítmica `abs(log2(tau_high / tau_low))` quando ambos os
valores existem. Ausência nunca é substituída por zero. Com
`allow_missing_tau=True`, a distância fica não aplicável; com
`allow_missing_tau=False`, o par é inadmissível.

Os testes cobrem tau semelhante, tau muito diferente, um ausente, ambos
ausentes, ausência permitida e ausência proibida.

## 8. Tracking

`CandidateReference` foi estendida com campos opcionais derivados do
`ModalCandidate`: estabilidade, drift, RMSE, cobertura, frações ambígua e
near-threshold, margem mínima e diagnósticos. A associação entre condições
reutiliza esses valores, sem recalcular tracking.

Os termos de tracking são opcionais. Quando habilitados, custos e gates ficam
explícitos no diagnóstico.

## 9. Pré-impacto

Evidência pré-impacto entra como compatibilidade operacional. Componentes
emergentes, amplificados e reexcitados são compatíveis quando ambos indicam
excitação por impacto. Linhas persistentes de fundo podem corresponder quando
o requisito de excitação não está habilitado.

Ausência de evidência é preservada como `None`. Com
`allow_missing_preimpact_evidence=False` ou `require_impact_excitation=True`,
candidatos sem evidência obrigatória são preservados sem match.

## 10. Ambiguidade

O matching é húngaro, determinístico e um-a-um. Alternativas inadmissíveis são
excluídas da matriz de custo. Margens de linha, coluna e margem operacional
são públicas; `ambiguous=True` quando a margem é menor ou igual ao limiar.

Exemplo real:

```text
pp: 100.0 Hz
p: 99.9 Hz e 100.1 Hz
custos = 0.05 e 0.05
margem = 0.0
match selecionado por desempate determinístico
alternativa preservada como emergente por ambiguidade
```

## 11. Candidatos emergentes

`EmergingCandidate` preserva candidatos da condição superior sem par. O
resultado registra referência, condição, melhor alternativa, menor custo
observado, razão e diagnósticos.

No teste básico:

```text
p: 450.0 Hz
razão = no_candidate_in_frequency_range
```

## 12. Candidatos desaparecidos

`DisappearingCandidate` preserva candidatos da condição inferior sem par.

No teste básico:

```text
pp: 300.0 Hz
razão = no_candidate_in_frequency_range
```

## 13. Possíveis splits

`PossibleCandidateSplit` é apenas diagnóstico. Ele aparece quando um candidato
inferior possui duas ou mais alternativas superiores admissíveis e próximas,
sem dominante clara.

Exemplo real:

```text
p: 200.0 Hz
mf: 198.5 Hz e 201.5 Hz
custos = 0.75 e 0.75
possible_split = True
```

O matching principal continua um-a-um.

## 14. Possíveis merges

`PossibleCandidateMerge` é análogo e também apenas diagnóstico.

Exemplo real:

```text
mf: 298.5 Hz e 301.5 Hz
f: 300.0 Hz
custos = 0.75 e 0.75
possible_merge = True
```

O candidato inferior restante é preservado sem fusão automática.

## 15. Determinismo

As referências são ordenadas por rótulo, frequência, gravação, candidato e
track. Entradas ordenadas, invertidas e embaralhadas produziram a mesma forma
normalizada de matches, IDs, custos, desaparecidos, emergentes, splits, merges
e diagnósticos.

## 16. Testes adicionados

Foram adicionados 59 testes em
`tests/test_cross_condition_candidate_association.py`, cobrindo:

- exemplo básico `pp -> p`;
- métricas frequenciais;
- deslocamento preservado, para cima e para baixo;
- limites inclusivos;
- ambiguidade e margens 0.05, 0.10 e 0.20;
- split e merge diagnósticos;
- tau;
- pré-impacto;
- candidatos rejeitados;
- pares inválidos;
- determinismo;
- perturbação local;
- ausência de correspondência confiável;
- dados insuficientes;
- invariantes de configuração.

## 17. Resultado final

Validação final da rodada:

```text
pytest
772 passed

pytest -W error
772 passed

python3 -m compileall -q belllab tests
OK

git diff --check
OK
```

## 18. Limitações

- Não há associação direta `pp -> ff` como política principal.
- Não há agrupamento global em todas as condições.
- Não há `ModalMode`.
- Não há famílias modais.
- Não há fator Q.
- Não há prova de não linearidade.
- Não há transição formal de regime.
- Não há classificação causal.
- Não há machine learning.
- Não há visualização final.
- Split e merge não são resolvidos automaticamente.

## 19. Próximos passos

- Aplicar a associação adjacente a conjuntos reais de candidatos por condição.
- Refinar calibração dos limiares apenas com validação experimental explícita.
- Projetar, em etapa futura separada, contratos para famílias modais ou modos
  físicos sem retroagir essa interpretação para a camada operacional atual.
