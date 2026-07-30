# Development report: reproducible scientific report v1

## 1. Data

2026-07-30.

## 2. Branch

`feature/reproducible-scientific-report`.

## 3. Estado inicial

Antes de qualquer edição, a branch foi confirmada como
`feature/reproducible-scientific-report`, diferente de `main`, com árvore de
trabalho limpa. A validação inicial executou:

- `pytest`: 1307 testes aprovados;
- `pytest -W error`: 1307 testes aprovados.

O arquivo `belllab/models.py` solicitado não existe nesta branch. Os contratos
equivalentes foram reutilizados de `belllab/types.py` e dos testes de modelos.

## 4. Arquivos criados

- `belllab/scientific_report.py`;
- `tests/test_scientific_report.py`;
- `examples/create_scientific_report.py`;
- `docs/development-report-reproducible-scientific-report-v1.md`.

## 5. Arquivos alterados

- `README.md`;
- `docs/RFC-0001-scientific-specification.md`;
- `belllab/__init__.py`.

## 6. Princípio científico

A camada preserva explicitamente:

```text
relatorio compilado != conclusao cientifica comprovada
figura incluida != evidencia adicional
tabela completa != dados suficientes
pipeline concluido != experimento fisicamente valido
hipotese modal != modo fisico comprovado
associacao entre condicoes != prova de identidade fisica
trajetoria de frequencia != prova de nao linearidade
anticorrelacao temporal != transferencia fisica comprovada
validacao sintetica != validacao experimental universal
```

O relatório distingue dado medido, estimativa, resultado operacional,
hipótese, ressalva, inconclusão, insuficiência, invalidez e limitação. Texto
automático é factual; discussão ou conclusão física só pode vir como texto
explícito do usuário, marcado como `user_provided_text`.

## 7. Arquitetura

`belllab.scientific_report` recebe `ExperimentAnalysisResult`,
opcionalmente `ExperimentExportResult` e opcionalmente
`ScientificFigureCollection`. Ele reutiliza `normalize_experiment_for_export`
para montar tabelas a partir de resultados existentes, reutiliza checksums e
políticas de overwrite da camada de exportação, e reutiliza artefatos de
figuras sem regenerá-los.

## 8. Configuração

`ScientificReportSettings` controla formatos, seções, conteúdo, figuras,
tabelas, texto, LaTeX, arquivos, numérico, overwrite e escrita atômica. Os
defaults são conservadores: PDF desligado, `shell_escape=False`, caminhos
relativos, validação de checksums habilitada, autores e financiamento não
inventados, e ausências preservadas.

## 9. Modelo normalizado

`ScientificReportDocument` é independente de Markdown e LaTeX. Ele contém
`report_id`, `analysis_id`, `experiment_id`, título, autores, idioma, seções,
figuras, tabelas, apêndices, referências cruzadas, proveniência, limitações,
diagnósticos, fingerprint de configuração e validade.

## 10. Seções

A estrutura padrão cobre capa, resumo, experimento, aquisição, metodologia,
qualidade, temporal, espectral, tracking, candidatos, associações, cadeias,
hipóteses, parâmetros, Q, energia operacional, validação sintética, síntese
factual, limitações, proveniência e apêndices. Cada seção pode ser desligada.

## 11. Narrativa conservadora

O renderizador bloqueia frases proibidas como "confirmed energy transfer" e
"modo fisico confirmado". Frases automáticas usam termos como hipótese modal,
evidência operacional, possível redistribuição, slope positivo ou negativo e
resultado inconclusivo. O relatório não declara causalidade, identidade modal
física ou prova de não linearidade.

## 12. Markdown

`render_scientific_report_markdown(...)` produz UTF-8, hierarquia de títulos,
tabelas, links relativos para figuras, equações, avisos científicos, apêndices
e metadado de schema. HTML bruto não é inserido; caracteres sensíveis são
sanitizados em links e células.

## 13. LaTeX

`render_scientific_report_latex(...)` produz documento compilável com
`booktabs`, `longtable`, `graphicx`, `hyperref`, `amsmath`, labels estáveis e
caminhos relativos. Caracteres especiais em texto do usuário, nomes e tabelas
são escapados.

## 14. PDF opcional

`compile_scientific_report_pdf(...)` suporta `latexmk`, `tectonic`, `pdflatex`,
`lualatex` e `xelatex` via detecção local. A função não instala ferramentas,
não acessa rede, executa no diretório do relatório, registra comando, logs,
stdout/stderr resumidos, warnings, erros e artefatos auxiliares. Ausência de
compilador gera status estruturado quando PDF é opcional.

## 15. Tabelas

`ScientificReportTable` reutiliza as tabelas normalizadas existentes:
resumo, gravações, condições, candidatos, associações, chains, hipóteses,
parâmetros, Q, energia operacional, estágios e diagnósticos. `None` é
renderizado segundo política explícita, nunca como zero.

## 16. Figuras

`ScientificReportFigureReference` aponta para artefatos já gerados por
`ScientificFigureCollection`. O relatório valida `analysis_id`, `experiment_id`
e checksums, usa caminhos relativos a partir do diretório do relatório e cria
legendas conservadoras. Figuras não são alteradas nem regeneradas.

## 17. Referências cruzadas

Labels determinísticos são criados para seções, tabelas, figuras e apêndices.
`validate_scientific_report(...)` confere unicidade e consistência entre o
inventário e as referências.

## 18. Metodologia

A metodologia é derivada da versão BellLab, estágios, configurações e
convenções existentes. Quando aplicável, o relatório mostra:

```text
A(t) = A0 exp(-t/tau)
Q_decay = pi f tau
Q_bandwidth = f0 / Delta_f
```

Essas equações são explicitamente descritas como convenções operacionais.

## 19. Parâmetros

A seção de parâmetros relata frequência representativa, trajetória, slope
ordinal, drift, tau, taxa de decaimento, incertezas, status, razões e
proveniência. Slopes não são convertidos em hardening, softening ou prova de
não linearidade.

## 20. Q

A seção de Q e largura de banda preserva Q por decaimento, Q por bandwidth,
Q representativo, bandwidth, resolução, isolamento, discordância entre métodos,
status e ressalvas. Um método discordante não é ocultado.

## 21. Energia operacional

O título automático é conservador:

```text
Evidencia operacional de possivel redistribuicao entre componentes
```

O relatório pode incluir envelopes, proxy de energia, tendência, correlação,
lag, crescimento tardio, recuperação, alternância, possível batimento, score e
status, sem afirmar transferência física.

## 22. Validação sintética

A seção de validação sintética declara que desempenho em sinais sintéticos não
garante desempenho equivalente em gravações reais. Resultados sintéticos são
apresentados como validação operacional controlada.

## 23. Limitações

Limitações são derivadas de status da análise, estágios omitidos, resultados
parciais, Q inconclusivo, figuras ausentes, export ausente, condições ausentes
e limitações gerais da RFC. Resultados negativos ou inválidos não são
descartados para melhorar a narrativa.

## 24. Proveniência

O documento preserva versão BellLab, analysis ID, experiment ID, fingerprint de
configuração da análise, fingerprint de configuração do relatório, hashes dos
arquivos de origem, export ID, figure collection ID, status das fontes e schema
do relatório.

## 25. Manifesto

`ScientificReportManifest` registra schema, IDs, versão BellLab, fingerprints,
seções, tabelas, figuras, apêndices, artefatos, checksums, compilação, status
das fontes, limitações e diagnósticos. O manifesto é escrito por último com
checksum próprio omitido para evitar identidade recursiva.

## 26. Checksums

`scientific_report_artifact_checksum(...)` usa SHA-256 por conteúdo. O relatório
valida checksums de export e figuras antes do render quando configurado.
Qualquer mismatch vira falha estruturada, não inserção silenciosa.

## 27. Relatório parcial

A camada aceita análise sem export, sem figuras, export parcial ou figuras
parciais. Markdown e LaTeX ainda podem ser gerados com o que existe, mas as
seções ausentes e limitações ficam registradas. A origem parcial não é
apresentada como completa.

## 28. Overwrite

São suportadas políticas `error`, `skip`, `replace` e `versioned_filename`.
Conflitos produzem falha ou artefato parcial conforme política; não há
overwrite silencioso.

## 29. Escrita atômica

Quando `atomic_write=True`, o conteúdo é escrito em temporário no mesmo
diretório, `fsync` é chamado, e `os.replace` conclui a troca. Falhas removem o
temporário e preservam o arquivo anterior.

## 30. Determinismo

O `report_id` depende de análise, export, coleção de figuras, seções, tabelas,
checksums e fingerprint de configuração, mas não do diretório de saída. Builds
em diretórios diferentes produzem estrutura e conteúdo normalizados estáveis.

## 31. Imutabilidade

Análise, export, figuras, metadados e textos do usuário não são modificados.
Listas são convertidas para tuplas, mapas para `MappingProxyType`, e não há RNG
global ou cache mutável global.

## 32. Testes

Foram adicionados 40 testes em `tests/test_scientific_report.py`, cobrindo:
contratos públicos, configuração inválida, fingerprint, documento normalizado,
seções, Markdown, LaTeX, manifesto, checksums, relatório parcial, IDs
incompatíveis, checksum mismatch, metadados ausentes, texto do usuário,
narrativa conservadora, referências cruzadas, PDF opcional, artefatos auxiliares,
overwrite, escrita atômica, determinismo, perturbação local, imutabilidade e
ausência de substituição de `None` por zero.

## 33. Resultado final

A suíte focal executou com `40 passed`. A validação completa da rodada deve
elevar a suíte de 1307 para 1347 testes aprovados. O exemplo executável criou
Markdown, LaTeX, Makefile, `.latexmkrc`, manifesto e, quando `latexmk` local
estava disponível, PDF compilado com checksum válido.

Exemplo quantitativo usado: cinco gravações sintéticas temporárias de 4096 Hz e
1,2 s com componentes primários `300, 301, 302, 303, 304 Hz` e secundários
`520, 521, 522, 523, 524 Hz`. O relatório organiza candidatos, hipóteses,
parâmetros, Q por decaimento e evidência operacional de energia sem criar
conclusões físicas.

## 34. Limitações da rodada

Não foram implementados interpretação física automática, discussão científica
automática, conclusão científica automática, submissão a periódico, template de
periódico, DOI, Zenodo, upload, publicação, servidor, GUI, dashboard, notebook
final, banco de dados, inferência causal, prova de não linearidade,
identificação física definitiva, resolução de split/merge, fechamento de
lacunas, associação não adjacente, elementos finitos ou mapas modais físicos.

## 35. Próximos passos

Rodadas futuras podem adicionar templates institucionais, empacotamento de
fontes do relatório, estilos de periódico configuráveis, integração com
referências bibliográficas reais e seleção assistida de apêndices. Essas
extensões devem manter o relatório como organização reproduzível, não como
prova científica autônoma.
