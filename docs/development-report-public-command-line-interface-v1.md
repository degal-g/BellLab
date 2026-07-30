# Development report: public command-line interface v1

## 1. Data

2026-07-30.

## 2. Branch

`feature/public-command-line-interface`.

## 3. Estado inicial

Antes de qualquer edição, a branch foi confirmada como
`feature/public-command-line-interface`, diferente de `main`, com árvore de
trabalho limpa. A validação inicial executou:

- `pytest`: 1347 testes aprovados;
- `pytest -W error`: 1347 testes aprovados.

O arquivo `belllab/models.py` solicitado não existe nesta branch; os contratos
canônicos equivalentes foram reutilizados de `belllab/types.py` e dos testes de
modelos.

## 4. Arquivos criados

- `belllab/cli.py`;
- `belllab/__main__.py`;
- `tests/test_cli.py`;
- `examples/experiment.example.toml`;
- `examples/experiment.example.json`;
- `examples/run_cli_workflow.sh`;
- `examples/run_cli_workflow.py`;
- `docs/development-report-public-command-line-interface-v1.md`.

## 5. Arquivos alterados

- `README.md`;
- `docs/RFC-0001-scientific-specification.md`;
- `belllab/__init__.py`;
- `pyproject.toml`.

## 6. Princípio científico

A CLI preserva explicitamente que comando concluído não é conclusão científica,
exit code `0` não elimina ressalvas científicas, configuração padrão não é
universal, relatório gerado não é prova física, hipótese modal não é modo
físico comprovado e evidência operacional não é causalidade nem transferência
física comprovada.

## 7. Arquitetura

`belllab.cli` é uma camada fina sobre APIs públicas existentes. `analyze` chama
`analyze_experiment` ou `resume_experiment_analysis`; `export` chama
`export_experiment_results`; `visualize` chama `create_experiment_visualizations`;
`report` chama `create_scientific_report`; `validate-synthetic` chama as APIs de
validação sintética; `inspect` apenas lê artefatos; `version` consulta a versão
canônica do pacote.

## 8. Entry point

`belllab/__main__.py` chama somente `belllab.cli.main`. O `pyproject.toml`
define:

```toml
[project.scripts]
belllab = "belllab.cli:main"
```

Assim, a interface funciona como `belllab --help` quando instalada e como
`python3 -m belllab --help` no checkout.

## 9. Parser

O parser usa `argparse` da biblioteca padrão. Não foram adicionadas dependências
como Click ou Typer. A ajuda principal lista subcomandos e cada subcomando
declara opções, escolhas permitidas e exemplos resumidos.

## 10. Comandos

Foram implementados:

- `belllab analyze`;
- `belllab export`;
- `belllab visualize`;
- `belllab report`;
- `belllab validate-synthetic`;
- `belllab inspect`;
- `belllab version`.

## 11. Configuração

`load_cli_configuration` lê JSON e TOML. Em Python sem `tomllib`, a CLI inclui
um parser TOML conservador suficiente para os exemplos públicos. Chaves
desconhecidas são rejeitadas por padrão.

## 12. Precedência

A precedência é:

```text
defaults < arquivo de configuração < opções da linha de comando
```

`merge_cli_configuration` é determinístico e não altera os mapas de entrada.

## 13. Analyze

`analyze` constrói `ExperimentDefinition` a partir de JSON/TOML ou de specs
rápidas `LABEL=PATH` e `LABEL:TAKE=PATH`. Ele mapeia opções de estágios para
`ExperimentPipelineSettings` e não duplica o grafo científico. `--until-stage`,
`--skip-stage` e `--only-stage` são validados pelas dependências canônicas do
pipeline.

Exemplo:

```bash
belllab analyze --config examples/experiment.example.toml --dry-run
```

## 14. Export

`export` carrega um bundle de análise BellLab já calculado e chama
`export_experiment_results`. Ele suporta JSON, CSV, LaTeX, Markdown, manifesto,
overwrite explícito, política de valores ausentes, política de não finitos,
precisão e validação de artefatos.

## 15. Visualize

`visualize` recebe um bundle de análise e chama
`create_experiment_visualizations`. Ele suporta seleção de figuras, `--all`,
PNG, SVG, PDF, DPI, estilos de resultados rejeitados/inconclusivos, IDs e modo
monocromático. Não abre janela gráfica e não recalcula FFT, STFT ou tracking.

## 16. Report

`report` recebe análise, export e coleção de figuras já gerados, e chama
`create_scientific_report`. Markdown e LaTeX são suportados; PDF é opcional.
Autor, instituição, título e idioma só aparecem quando fornecidos
explicitamente.

## 17. Validate-synthetic

`validate-synthetic` usa os cenários sintéticos públicos como `single_ideal`,
`beating`, `noise`, `energy_exchange` e `all-scenarios`. Seeds, número de
trials, SNR, clipping, sample rate e duração são explícitos. A verdade sintética
não é usada para calibrar thresholds ou reparar tracking.

## 18. Inspect

`inspect` reconhece bundles CLI, manifestos de exportação, manifestos de
relatório, export normalizado e configurações. Ele calcula checksum do arquivo
inspecionado e não modifica o conteúdo.

## 19. Version

`version` mostra versão BellLab, schema de exportação, schema de relatório e
schema dos bundles CLI. Com `--verbose`, inclui Python e plataforma.

## 20. Stdout

Saída humana é estável e curta. Saída JSON usa stdout apenas para JSON válido.
Help do `argparse` não recebe resumo adicional da CLI.

## 21. Stderr

Warnings e erros esperados são enviados para stderr pela função `main`. Erros
esperados não imprimem traceback.

## 22. JSON

`format_cli_json_output` serializa `BellLabCLIResult` sem `NaN` ou `Infinity`.
O payload contém command, exit code, status, IDs, paths, warnings, errors,
diagnostics e resumo estruturado.

## 23. Quiet

`--quiet` suprime saída normal, preservando exit code e permitindo mensagens de
erro fatal em stderr.

## 24. Logging

A CLI usa `logging` padrão com `--verbose`, `--debug`, `--log-file` e
`--log-level`. Handlers próprios são substituídos em execuções repetidas para
evitar acúmulo global.

## 25. Exit codes

Os códigos públicos são:

- `0`: concluído;
- `1`: concluído com ressalvas;
- `2`: uso ou configuração inválida;
- `3`: input inválido;
- `4`: evidência insuficiente;
- `5`: execução parcial;
- `6`: falha de estágio;
- `7`: erro interno inesperado;
- `8`: validação de artefato falhou;
- `9`: compilação de relatório falhou.

## 26. Dry-run

`--dry-run` valida argumentos e configuração, mostra artefatos previstos e não
executa análise pesada nem escreve arquivos. Para `analyze`, a validação de
entrada é executada sem carregar WAVs.

## 27. Resume

`analyze --resume-from result.json` carrega um bundle de análise BellLab e chama
`resume_experiment_analysis`, que reutiliza resultados pré-calculados segundo
as validações canônicas existentes.

## 28. Paths

Paths relativos, absolutos, `~`, espaços e UTF-8 são processados com `pathlib`.
Caminhos relativos em configuração são resolvidos relativamente ao arquivo de
configuração.

## 29. Privacidade

`--redact-paths` remove paths absolutos da saída da CLI, preservando IDs e
fingerprints. Os arquivos produzidos não são alterados por essa opção.

## 30. Determinismo

Fingerprints de configuração são baseados em serialização canônica. Bundles
registram schema, versão, summary, checksum do payload e payload reconstruível.
IDs científicos vêm das camadas canônicas.

## 31. Imutabilidade

`argv`, configurações e objetos BellLab não são modificados in-place. Os
arquivos de áudio de entrada não são alterados. Não há cache mutável global.

## 32. Testes

Foram adicionados 51 testes em `tests/test_cli.py`, cobrindo contratos públicos,
parser, help, version, config JSON/TOML, precedência, geração de config, specs
de gravação, dry-run, analyze, resume, export, visualize, report,
validate-synthetic, inspect, exit codes, stdout/stderr, JSON, quiet, privacidade,
determinismo, perturbação local, imutabilidade, logging e subprocessos com
`python3 -m belllab`.

## 33. Resultado final

A suíte focal executou com `51 passed`. A validação completa da rodada elevou a
suíte de 1347 para 1398 testes aprovados em `pytest` e `pytest -W error`.

Exemplos reais exercitados:

```bash
python3 -m belllab --help
python3 -m belllab version
python3 -m belllab analyze --recording pp=/tmp/pp.wav --until-stage global_spectrum --save-result analysis.json
python3 -m belllab export --analysis analysis.json --json --csv --output-dir export
python3 -m belllab visualize --analysis analysis.json --figure global_spectrum --format png --output-dir figures
python3 -m belllab report --analysis analysis.json --markdown --latex --output-dir report
```

## 34. Limitações

Não foram implementados GUI, TUI interativa, servidor, API HTTP, upload, rede,
publicação, DOI, Zenodo, banco de dados, cache persistente, execução
distribuída, scheduler, monitoramento em tempo real, gravação de áudio,
streaming, shell completion avançado, wizard interativo, inferência física
automática, discussão científica automática, resolução de split/merge,
associação não adjacente ou fechamento de lacunas.

## 35. Próximos passos

Rodadas futuras podem adicionar shell completion, reconstrução JSON completa
sem payload Python confiável, empacotamento de bundles, perfis de configuração
nomeados e integração com instaladores. Essas extensões devem manter a CLI como
adaptador às APIs públicas, não como nova camada de inferência.
