# Development report: reproducible scientific visualizations v1

## 1. Data

2026-07-29.

## 2. Branch

`feature/reproducible-scientific-visualizations`.

## 3. Estado inicial

Antes de qualquer alteração, a branch foi confirmada com
`git branch --show-current`, a árvore de trabalho estava limpa e a suíte base
executou com:

- `pytest`: 1253 testes aprovados;
- `pytest -W error`: 1253 testes aprovados.

Os arquivos `belllab/models.py`, `belllab/stft.py` e `belllab/peaks.py`
solicitados na leitura inicial não existem nesta branch. Os contratos canônicos
correspondentes foram lidos nos módulos existentes, principalmente
`belllab/types.py`, `belllab/spectrum.py`, `belllab/time_resolved_spectrum.py`
e nos testes de modelos, STFT e picos.

## 4. Arquivos criados

- `belllab/scientific_visualizations.py`;
- `tests/test_scientific_visualizations.py`;
- `examples/create_experiment_visualizations.py`;
- `docs/development-report-reproducible-scientific-visualizations-v1.md`.

## 5. Arquivos alterados

- `README.md`;
- `docs/RFC-0001-scientific-specification.md`;
- `belllab/__init__.py`;
- `requirements.txt`;
- `pyproject.toml`.

## 6. Princípio científico

A camada representa resultados operacionais já calculados. Ela preserva:

```text
figura visualmente convincente != evidência científica suficiente
linha ligando pontos != continuidade física comprovada
cor semelhante != identidade modal física
anticorrelação visual != transferência física de energia
trajetória de frequência != prova de hardening ou softening
diferença entre condições != prova de não linearidade
hipótese modal != modo físico comprovado
```

Nenhuma função de visualização reabre WAVs ou recalcula FFT, STFT, tracking,
candidatos, associações, parâmetros, Q ou evidência operacional de energia.

## 7. Arquitetura

O módulo `belllab.scientific_visualizations` recebe objetos BellLab existentes:
`Signal`, `Envelope`, `Spectrum`, resultados de gravação, resultados de
experimento, estimativas modais, Q, largura de banda, evidência operacional de
energia e validação sintética. Quando útil, ele reutiliza a normalização da
camada `results_export` para tabelas internas de candidatos, associações,
cadeias, hipóteses, parâmetros, Q e energia.

## 8. Configuração

`ScientificVisualizationSettings` centraliza saída, formatos, DPI, dimensões,
tipografia, linhas, escalas, seleção de camadas, decimação visual, paleta,
estilos, anotações, overwrite e escrita atômica. Os defaults são conservadores:
backend headless, `png` por padrão, overwrite `error`, fundo não transparente,
paleta determinística e sem escolhas aleatórias.

## 9. Backend

Matplotlib é usado com backend não interativo `Agg`. As APIs criam `Figure` e
`Axes` diretamente, não chamam `plt.show()` e usam `rc_context` local para
evitar mutação global persistente de `matplotlib.rcParams`.

## 10. Paleta

`ScientificColorPolicy` suporta `dynamic_condition`, `status`, `stable_hash`,
`monochrome` e `custom`. Para dinâmica, o mapeamento canônico é estável em:
`pp`, `p`, `mf`, `f`, `ff`. A codificação também usa marcadores e estilos,
evitando depender apenas de cor.

## 11. Status

`ScientificVisualizationStatus` define estados mutuamente exclusivos:
`created`, `created_with_reservations`, `skipped`, `insufficient_evidence`,
`failed` e `invalid_input`. `ScientificVisualizationReason` separa suporte,
ressalvas, insuficiência e falhas de renderização.

## 12. Proveniência

`ScientificFigureProvenance` registra `figure_id`, tipo, `analysis_id`,
`experiment_id`, gravações, condições, IDs de origem, fingerprint de
configuração, versão BellLab, status de origem, interpolações, decimações e
diagnósticos. IDs são determinísticos e não usam timestamp, UUID ou contador
global.

## 13. Waveform

`plot_waveform(...)` renderiza sinais já carregados, com eixo temporal,
unidade de amplitude, canais explícitos, limiar de clipping quando há evidência
de full-scale recorrente e decimação min-max para sinais longos. Não há downmix
nem normalização silenciosa.

Exemplo quantitativo: no exemplo executável, cinco WAVs temporários de
4096 Hz e 1,2 s são analisados com componentes em 300-304 Hz e 520-524 Hz.

## 14. Envelope

`plot_temporal_envelope(...)` renderiza `Envelope` ou resultados temporais
existentes em escala linear ou dB. Zeros em escala log são recortados somente
para apresentação e geram diagnóstico `log_scale_clipped`.

## 15. Decaimento

`plot_decay_estimate(...)` mostra envelopes e ajustes de decaimento já
disponíveis, tau operacional, janelas de ajuste e parâmetros modais quando
existem. A função não reestima tau nem ajusta nova curva.

## 16. Espectro

`plot_global_spectrum(...)` recebe `Spectrum` ou resultados espectrais
existentes, suporta frequência linear/log e amplitude linear/dB, picos
disponíveis, piso dB e anotações limitadas. Nenhuma FFT é recalculada.

## 17. Espectrograma

`plot_spectrogram(...)` renderiza `TimeFrequencySpectrum` existente com barra
de cores e convenção de magnitude ou dB. Matrizes ausentes ou inconsistentes
resultam em insuficiência ou entrada inválida, não em dados fabricados.

## 18. Tracks

`plot_frequency_tracks(...)` desenha tracks existentes no plano tempo versus
frequência, com pontos e segmentos por track. Gaps não são ligados, tracks
diferentes não são conectados e diagnósticos deixam claro que tracking não foi
refeito.

## 19. Candidatos

`plot_modal_candidates(...)` usa frequências representativas e status de
candidatos normalizados. Marcadores e alpha preservam aceitação, rejeição,
ressalvas e insuficiências. O título usa candidatos, não modos físicos.

## 20. Associações

`plot_within_condition_associations(...)` e
`plot_cross_condition_associations(...)` desenham nós e arestas operacionais de
associação. Condições não adjacentes não recebem arestas e lacunas permanecem
diagnosticadas. Split e merge podem aparecer apenas como contexto.

## 21. Chains

`plot_candidate_chains(...)` mostra cadeias em eixo ordinal de condição,
preservando início, término, lacunas e completude operacional. Cadeias não são
promovidas a modos físicos.

## 22. Hipóteses

`plot_modal_hypotheses(...)` renderiza hipóteses modais com status, cobertura,
frequência e ressalvas. O título declara explicitamente que não são modos
físicos comprovados.

## 23. Trajetórias

`plot_modal_frequency_trajectories(...)` mostra frequência versus condição,
incerteza quando disponível e slope ordinal existente. O diagnóstico declara
que a trajetória não é prova de hardening, softening ou não linearidade.

## 24. Parâmetros

`plot_modal_parameters(...)` cria painéis para frequência representativa, tau e
taxa de decaimento, preservando `None` como ausência. A visualização não combina
parâmetros ausentes.

## 25. Q

`plot_modal_q_factors(...)` mostra Q por decaimento, Q por largura de banda e Q
representativo quando disponíveis, com incertezas e status. Discordâncias entre
métodos não são ocultadas.

## 26. Bandwidth

`plot_modal_bandwidth(...)` mostra centro, cruzamentos inferior/superior,
largura e resolução quando a largura já foi estimada. Largura ausente não é
representada como zero.

## 27. Condições dinâmicas

`plot_dynamic_condition_comparison(...)` usa posições gráficas ordinais para
`pp`, `p`, `mf`, `f`, `ff`. O espaçamento não é tratado como intensidade física
calibrada.

## 28. Energia operacional

`plot_modal_energy_exchange_evidence(...)` mostra envelopes, proxy operacional
de energia do par, score, status, tendências e ressalvas.
`plot_modal_energy_exchange_correlation(...)` mostra correlação versus lag e a
convenção de sinal do lag sem declarar direção causal.

## 29. Validação sintética

`plot_synthetic_validation_result(...)` e
`plot_synthetic_validation_campaign(...)` mostram verdadeiro versus estimado,
erros, status, campanhas e Monte Carlo. A figura declara que sucesso sintético
não prova validade geral em dados reais.

## 30. Decimação

A decimação é visual e determinística. Ela preserva primeiro e último ponto e,
quando configurada, máximos e mínimos por bloco. A quantidade original e a
quantidade renderizada ficam na proveniência.

## 31. Anotações

Anotações têm limite determinístico. Quando há excesso, a camada reduz textos e
registra ressalva; dados não são removidos, apenas rótulos podem ser omitidos.

## 32. Escrita de arquivos

`save_scientific_figure(...)` salva PNG, SVG e PDF quando suportados pelo
backend, com caminhos relativos, checksums SHA-256 de conteúdo, overwrite
explícito e escrita atômica opcional. A função reutiliza a política de
overwrite da camada de exportação.

## 33. Determinismo

Fingerprints excluem diretórios de saída. IDs de figuras dependem de tipo,
fontes e configuração científica, não de caminho de exportação. A paleta e a
ordem de renderização são estáveis sob ordens equivalentes de entrada.

## 34. Imutabilidade

As funções convertem entradas para arrays temporários e não reordenam listas
in-place. Figuras são fechadas após salvamento quando `close_after_save=True`.
O RNG global não é usado e arquivos de áudio de origem não são modificados.

## 35. Testes

Foi adicionada a suíte `tests/test_scientific_visualizations.py`, com 54 testes
cobrindo contratos públicos, configuração inválida, fingerprints, paleta,
status, waveform, envelope, decaimento, espectro, picos, espectrograma, tracks,
candidatos, associações, cadeias, hipóteses, trajetórias, parâmetros, Q,
bandwidth, energia operacional, correlação e lag, validação sintética,
coleções, PNG/SVG/PDF, overwrite, escrita atômica, determinismo, perturbação
local, imutabilidade, `rcParams` e fechamento de figuras.

## 36. Resultado final

A validação focal executou com `54 passed`. A validação completa da rodada deve
preservar os 1253 testes iniciais e elevar a suíte para 1307 testes aprovados.
O exemplo `examples/create_experiment_visualizations.py` gerou 32 figuras e 64
artefatos PNG/SVG com checksums válidos.

## 37. Limitações

Esta rodada não implementa relatório PDF final, HTML final, dashboard, GUI,
notebook final, animações, vídeo, gráficos 3D, mapas modais físicos, mesh,
elementos finitos, inferência causal, prova de não linearidade, identificação
física definitiva, resolução de split ou merge, fechamento de lacunas,
associação não adjacente, recalibração automática, escolha automática de
conclusões ou publicação automática.

## 38. Próximos passos

Próximas rodadas podem adicionar manifesto visual agregado, presets de layout
para artigos, seleção assistida de painéis, figuras compostas por campanha e
integração com exportação textual final. Essas extensões devem continuar
preservando que figuras são representações de resultados operacionais, não nova
prova física.
