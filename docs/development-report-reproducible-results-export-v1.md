# Development Report: Reproducible Results Export v1

## 1. Data

2026-07-29.

## 2. Branch

`feature/reproducible-results-export`.

## 3. Estado Inicial

Antes de qualquer alteração foram confirmados:

- branch atual: `feature/reproducible-results-export`;
- branch diferente de `main`;
- árvore de trabalho limpa;
- `pytest`: 1204 testes aprovados;
- `pytest -W error`: 1204 testes aprovados.

## 4. Arquivos Criados

- `belllab/results_export.py`;
- `tests/test_results_export.py`;
- `docs/development-report-reproducible-results-export-v1.md`;
- `examples/export_experiment_results.py`.

## 5. Arquivos Alterados

- `README.md`;
- `docs/RFC-0001-scientific-specification.md`;
- `belllab/__init__.py`.

## 6. Princípio Científico

A camada de exportação preserva explicitamente:

```text
exportação bem-sucedida
≠ resultado cientificamente válido

tabela formatada
≠ evidência física suficiente

valor ausente
≠ zero

hipótese modal
≠ modo físico comprovado

evidência operacional de possível redistribuição
≠ transferência física comprovada

arquivo reproduzível
≠ experimento reproduzido fisicamente
```

A implementação exporta resultados já calculados. Ela não reabre WAV para
análise, não recalcula espectros, STFT, tracking, candidatos, hipóteses,
parâmetros, Q ou evidência operacional de energia.

## 7. Escopo

O escopo é a serialização determinística de `ExperimentAnalysisResult` e de seus
resultados intermediários. A camada preserva relações entre gravações,
condições, candidatos, cadeias, hipóteses, parâmetros, Q, evidência operacional
de possível redistribuição de energia, estágios, diagnósticos, configurações e
proveniência.

Não foram implementados PDF, HTML final, dashboard, figuras científicas finais,
relatório interpretativo completo, cache persistente ou reconstrução integral
dos dataclasses a partir do JSON.

## 8. Configuração

`ResultsExportSettings` controla formatos, conteúdo, numérico, arquivos,
tabelas e identidade. Os defaults são conservadores:

- exporta JSON, CSV, LaTeX, Markdown, resumo e manifesto;
- inclui resultados inválidos, rejeitados e inconclusivos;
- usa `overwrite_policy=error`;
- usa `nonfinite_value_policy=error`;
- preserva precisão completa no JSON;
- usa SHA-256 para checksums;
- usa escrita atômica.

Configurações contraditórias são rejeitadas, por exemplo desabilitar todos os
artefatos solicitáveis, precisão não positiva, checksum não suportado ou
política desconhecida.

## 9. Schema

`BellLabExportSchemaVersion.V1_0` identifica a primeira versão do schema de
exportação. A versão do schema é separada da versão do pacote para permitir
evolução futura. O significado de valores ausentes é preservado: `None` em
objetos Python e `null` em JSON indicam indisponibilidade real, não zero.

## 10. Normalização

`NormalizedExperimentExport` é a representação intermediária. Ela contém:

- versão do schema;
- versão do BellLab;
- `analysis_id`;
- `experiment_id`;
- definição do experimento;
- resumo;
- gravações;
- condições;
- resultado entre condições;
- cadeias;
- hipóteses modais;
- parâmetros;
- Q;
- evidência operacional de possível redistribuição;
- estágios;
- proveniência;
- configurações;
- diagnósticos;
- tabelas normalizadas.

Enums são convertidos para seus valores estáveis, dataclasses são convertidos
recursivamente, tuplas viram sequências serializáveis, dicionários são ordenados
por chave e as entradas originais não são modificadas.

## 11. JSON

`export_experiment_json(...)` escreve `experiment_export.json` com UTF-8,
indentação determinística, chaves ordenadas e `allow_nan=False`. O JSON preserva
valores completos por padrão. Um roundtrip estrutural:

```text
NormalizedExperimentExport → JSON → dict/list
```

é validado para IDs, status, contagens, `None`, diagnósticos e proveniência.

## 12. CSV

`export_experiment_csv_tables(...)` produz tabelas normalizadas:

```text
experiment_summary.csv
recordings.csv
conditions.csv
candidates.csv
within_condition_associations.csv
cross_condition_matches.csv
candidate_chains.csv
candidate_chain_nodes.csv
modal_hypotheses.csv
modal_parameters.csv
modal_q_factors.csv
energy_exchange_pairs.csv
pipeline_stages.csv
diagnostics.csv
```

As tabelas têm cabeçalhos estáveis, linhas determinísticas, IDs de origem,
status, razões e campos de incerteza quando disponíveis. Listas internas são
serializadas como JSON compacto dentro da célula, nunca como `repr` Python.

## 13. LaTeX

`export_experiment_latex_tables(...)` escreve fragmentos `.tex` separados para
resumo, gravações, hipóteses, parâmetros, Q, energia operacional e falhas ou
ressalvas. Os fragmentos usam `booktabs` quando configurado, escapam caracteres
especiais e não incluem preâmbulo completo.

Exemplo de valor de apresentação:

```text
representative_frequency_hz = 300.000000
representative_tau_s = -
```

O traço indica ausência na camada de apresentação, não zero.

## 14. Markdown

`export_experiment_markdown_summary(...)` cria um resumo estável com:

- identificação do experimento;
- status global;
- gravações;
- condições;
- estágios;
- contagens principais;
- hipóteses;
- parâmetros;
- Q;
- evidência operacional de possível redistribuição;
- falhas;
- ressalvas;
- proveniência;
- limitações.

O texto evita narrativa física conclusiva.

## 15. Valores Ausentes

`ExportMissingValuePolicy` define representação de ausência em CSV, LaTeX e
Markdown:

- `null`;
- célula vazia;
- `NA`;
- `-`.

JSON usa `null`. A camada nunca substitui `None` por `0`.

## 16. Não Finitos

`ExportNonfiniteValuePolicy` define:

- `error`, default;
- `null_with_diagnostic`;
- `string_with_diagnostic`.

JSON nunca escreve tokens não padronizados `NaN`, `Infinity` ou `-Infinity`.
Quando uma política permissiva é usada, o diagnóstico registra o caminho do
valor não finito.

## 17. Arredondamento

`ExportNumericFormatting` controla a apresentação em tabelas. O arredondamento
é aplicado apenas a CSV, LaTeX e Markdown. O modelo normalizado e o JSON
preservam precisão completa por padrão.

Exemplo quantitativo de teste:

```text
frequência sintética: 300.0 Hz
checksum JSON: SHA-256 de conteúdo
row_count de recordings.csv: 1
```

## 18. Manifesto

`ExperimentExportManifest` registra:

- versão do manifesto;
- versão do schema;
- `analysis_id`;
- `experiment_id`;
- versão do BellLab;
- fingerprint das configurações;
- fingerprints de arquivos de origem já presentes na proveniência;
- artefatos gerados;
- checksums;
- tamanhos;
- row counts;
- exportações concluídas, omitidas e falhas;
- status da fonte.

O manifesto é escrito por último. Para manter identidade portável, a lista de
artefatos no manifesto usa caminhos relativos.

## 19. Checksums

`export_artifact_checksum(...)` calcula SHA-256 do conteúdo escrito. O hash não
usa nome de arquivo ou caminho. O checksum é calculado depois da escrita
concluída.

## 20. Escrita Atômica

Quando `atomic_write=True`, a camada escreve primeiro um arquivo temporário no
mesmo diretório, faz flush e `fsync`, e só então substitui o destino. Em falha,
o temporário é removido e o destino anterior permanece preservado.

## 21. Overwrite

`ExportOverwritePolicy` define:

- `error`;
- `skip`;
- `replace`;
- `versioned_filename`.

O default é `error`. Em `versioned_filename`, o sufixo local é determinístico:
`_v001`, `_v002`, e assim por diante.

## 22. Exportação Parcial

Artefatos não solicitados são marcados como `skipped`, não como falha. A
exportação aceita resultados de pipeline completos, com ressalvas, parciais,
insuficientes ou falhos, preservando o status da origem e marcando a exportação
com ressalvas quando o resultado fonte requer revisão.

## 23. Validação

`ExperimentExportValidation` confere:

- artefatos esperados;
- arquivos existentes;
- checksums;
- roundtrip JSON;
- consistência do manifesto;
- relações básicas entre CSVs, como candidatos referenciando gravações
  exportadas.

Não é exigida reconstrução integral dos dataclasses científicos nesta versão.

## 24. Determinismo

A identidade da exportação usa `analysis_id`, `experiment_id`, fingerprint de
configuração e checksums de conteúdo. O diretório físico de saída é excluído do
fingerprint da configuração. Manifestos usam caminhos relativos para que a
mesma análise exportada em diretórios diferentes produza o mesmo conteúdo.

## 25. Imutabilidade

Os testes confirmam que:

- `ExperimentAnalysisResult` não é modificado;
- listas não são reordenadas in-place;
- metadados e diagnósticos não são convertidos in-place;
- arquivos de áudio fonte não são alterados;
- exportações repetidas são estáveis;
- não há cache global mutável.

## 26. Testes

Foram adicionados 49 testes cobrindo:

- contratos públicos;
- configuração;
- fingerprint;
- schema;
- normalização;
- JSON;
- CSV;
- LaTeX;
- Markdown;
- manifesto;
- checksums;
- escrita atômica;
- overwrite;
- valores ausentes;
- valores não finitos;
- resultados inválidos;
- exportação parcial;
- validação;
- determinismo;
- perturbação local;
- imutabilidade.

## 27. Resultado Final

Resultado esperado da rodada:

```text
pytest: 1253 passed
pytest -W error: 1253 passed
python3 -m compileall -q belllab tests examples: aprovado
git diff --check: aprovado
```

## 28. Limitações

Esta versão não implementa PDF, HTML final, dashboard, GUI, figuras, banco de
dados, cache persistente, upload, servidor, DOI, Zenodo, relatório
interpretativo completo, narrativa física automática, reconstrução total de
dataclasses pelo JSON, leitura de áudio para recalcular resultados ou alteração
dos resultados científicos.

## 29. Próximos Passos

- Definir um schema publicado em arquivo separado;
- ampliar validação de chaves estrangeiras entre todas as tabelas;
- adicionar exportação de dados para figuras futuras;
- implementar reconstrução parcial de objetos científicos a partir do JSON;
- integrar a exportação ao futuro relatório científico final sem inferir
  conclusões físicas automaticamente.
