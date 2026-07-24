# Development Report — Modal Candidate v1

**Data:** 2026-07-23  
**Estado inicial:** 234 testes aprovados.

## Escopo e definição científica

Esta rodada adicionou uma camada operacional explícita entre caracterização de
trajetória e qualquer interpretação modal futura. Um `ModalCandidate` é uma
trajetória espectral caracterizada que satisfez critérios matemáticos e
operacionais configuráveis. Ele não é um modo físico comprovado, frequência
natural validada, fator Q, amortecimento físico, forma ou energia modal.

Foram alterados `belllab/config.py`, `belllab/types.py`, `belllab/__init__.py`,
`README.md`, `docs/RFC-0001-scientific-specification.md` e adicionados
`belllab/modal_candidates.py`, `tests/test_modal_candidates.py` e este
relatório. A estrutura anterior `ModalMode` permaneceu intacta e não é
alimentada por esta camada.

## Contratos e API

`ModalCandidateSettings` contém critérios opcionais. `None` desabilita limites
numéricos; requisitos booleanos são habilitados apenas por `True`. O padrão
exige somente duas observações e frequência representativa válida, sem exigir
decaimento.

`CandidateCriterionResult` registra nome, valor observado, operador, limite,
habilitação, aplicabilidade, resultado e razão curta. `ModalCandidate` mantém a
caracterização como fonte canônica e expõe resumos por propriedades de leitura,
evitando cópias divergentes. Seus campos próprios preservam IDs, diagnóstico de
tracking, critérios, aceitação e razões.

As funções públicas são `evaluate_modal_candidate` e
`select_modal_candidates`. A seleção retorna todos os candidatos, aceitos e
rejeitados, em ordem de `source_track_id`; `candidate_id` é a posição
determinística nessa ordem.

## Frequência representativa

A frequência representativa operacional usa primeiro a mediana da série
canônica e a média somente como fallback. Ausência de ambas rejeita a
candidatura. A mediana reduz a influência de outliers e pequenas derivações,
mas não é chamada de frequência natural. No teste 100, 101 e 400 Hz, a
frequência representativa foi 101 Hz, não a média 200,3333333333 Hz.

## Critérios configuráveis

### Persistência

São independentes o mínimo de observações, a duração mínima e a cobertura
mínima. Uma trajetória com seis observações, duração 5 s e cobertura 1 foi
aceita pelos limites 5, 4 s e 0,9. Quadros 0 e 2 produziram cobertura 2/3 e
rejeição específica diante do mínimo 0,8.

### Estabilidade frequencial

Podem ser exigidos estabilidade relativa máxima, deriva absoluta máxima, RMSE
máximo e sucesso do `TrackFrequencyFit`. Origem `mixed` é aceita por padrão e
pode ser rejeitada explicitamente; origem `bin` não é rejeitada
automaticamente.

### Amplitude

Decaimento não é obrigatório por padrão. Quando configurados, são avaliados
decaimento detectado, R² mínimo e faixa de tau. A trajetória exponencial
`exp(-t)` recuperou tau 1 s e R² 1 e passou pela faixa de 0,5 a 2 s. Séries com
tau 0,2 s e 3 s foram rejeitadas separadamente pelos limites correspondentes.
R² indisponível falha apenas quando seu critério está habilitado.

### Qualidade do tracking

As frações ambígua e próxima do limite usam:

`count / accepted_assignment_count`

O denominador inclui somente diagnósticos públicos de associações aceitas para
o `track_id` avaliado. Duas associações ambíguas em três produziram 2/3 e
falharam diante do máximo 0,5; o mesmo cenário foi validado para
`near_threshold`.

A margem usada é o mínimo das margens públicas disponíveis. Margem 0,1 falhou
diante do mínimo 0,2. Quando todas as associações possuem alternativa única,
a margem permanece `None`; o critério é `not_applicable`, não rejeita e gera
`assignment_margin_not_applicable`.

Quando não há associação auditável, as frações permanecem `None` e não há
divisão por zero. Um limite de fração habilitado falha com valor indisponível;
desabilitado, ele não afeta a candidatura.

## Aceitos, rejeitados e determinismo

Candidatos rejeitados preservam caracterização, critérios estruturados, razões
e diagnósticos. Isso permite rever limiares sem repetir o tracking. Um candidato
aceito não possui critério habilitado reprovado nem razão de rejeição; um
rejeitado possui ao menos uma razão específica.

Seleções com duas trajetórias foram executadas duas vezes. Ordem, IDs `(0, 1)`,
frequências representativas, critérios, razões e diagnósticos foram iguais.
Alterar somente `minimum_duration_s` modificou somente esse resultado de
critério.

## Validação e invariantes

As configurações rejeitam contagens, durações, estabilidade, deriva, RMSE e
margens negativas; frações fora de `[0, 1]`; tau não positivo ou invertido; e
todo NaN ou infinito.

Os contratos rejeitam IDs incompatíveis, frequência representativa inválida,
contagens e frações incoerentes, critérios duplicados, textos vazios, estados
habilitado/aplicável contraditórios, candidato aceito com reprovação e candidato
rejeitado sem razão.

Foram adicionados **50 casos de teste**, elevando a suíte de **234 para 284
testes**.

- `pytest`: 284 aprovados;
- `pytest -W error`: 284 aprovados;
- `python3 -m compileall -q belllab tests`: aprovado;
- `git diff --check`: aprovado;
- não há ferramenta estática adicional configurada em `pyproject.toml`.

## Limitações e próximos passos

Os critérios são gates operacionais, não probabilidades ou confiança modal.
Mediana, estabilidade, RMSE, cobertura, margem e tau não comprovam identidade
física. Não há amplitude absoluta obrigatória porque não existe normalização
física universal nesta camada.

Uma promoção futura para `ModalMode` exigirá critérios físicos e validação
adicionais. Esta versão não implementa conversão automática ou manual,
fator Q, amortecimento físico, agrupamento harmônico, famílias modais, energia,
comparação, visualização ou exportação final.

## Contract closure

O fechamento das invariantes mantém os critérios científicos e seus padrões
inalterados. Um candidato aceito exige ausência de critérios habilitados e
aplicáveis reprovados, ausência de razões ou diagnósticos de rejeição e razões
de aceitação iguais, na ordem dos critérios, às razões dos critérios aprovados.
Critérios desabilitados ou não aplicáveis não impedem aceitação.

Um candidato rejeitado exige ao menos um critério habilitado, aplicável e
reprovado. Suas razões de rejeição são exatamente as razões desses critérios,
na mesma ordem; texto arbitrário, critério desabilitado ou critério não
aplicável não pode justificar rejeição. Ambas as coleções de razões são tuplas
de strings únicas e não vazias.

Falhas estruturais são representadas por critérios auditáveis. Em particular,
ausência de frequência válida reprova `representative_frequency_hz`; não existe
uma categoria livre de razões estruturais. `evaluate_modal_candidate` deriva as
duas coleções diretamente dos critérios e `select_modal_candidates` preserva
ordem e IDs determinísticos.

Foram adicionados 19 casos de fechamento. A suíte final possui **303 testes
aprovados** tanto com `pytest` quanto com `pytest -W error`; `compileall` e
`git diff --check` também foram aprovados.
