# Development Report — Spectral Tracking v1

**Data:** 2026-07-23  
**Estado inicial:** 85 testes aprovados.

## Escopo e organização

Esta entrega implementa detecção de picos em quadros STFT e associação temporal
de picos em trajetórias espectrais. A organização mínima escolhida foi adicionar
`belllab/tracking.py`; mover FFT, picos e STFT para módulos separados causaria
uma reorganização mais ampla sem benefício imediato. As APIs existentes em
`spectrum.py` foram preservadas.

Nenhuma trajetória é um `ModalMode`, nem recebe interpretação modal, fator Q,
amortecimento, energia modal ou classificação física.

## Contratos e detecção por quadro

Foram adicionados `FramePeakDetectionSettings`, `SpectralTrackingSettings`,
`FramePeaks`, `SpectralTrack`, `TimeFrequencyPeakResults` e
`SpectralTrackingResults`. `AnalysisSettings` agora agrega `frame_peaks` e
`tracking` sem substituir configurações específicas.

`detect_stft_peaks` recebe uma `TimeFrequencySpectrum` já calculada. Cada
coluna da matriz STFT é apresentada ao `detect_spectral_peaks` existente por
uma visão `Spectrum` compatível; portanto proeminência, largura, interpolação
sub-bin, piso local, SNR e filtros de frequência têm uma única implementação.
Quadros silenciosos ou abaixo do limiar podem ser ignorados como resultados
válidos, não como falhas.

## Associação e lacunas

`track_spectral_peaks` usa associação Húngara, determinística e um-para-um. O
custo inicial é a distância frequencial normalizada pela tolerância, ponderada
por `frequency_weight`; uma diferença de amplitude normalizada pode ser somada
com `amplitude_weight`. A distância é explicitamente uma de `hz`, `relative`
ou `cents`; elas nunca são misturadas automaticamente. A frequência refinada é
preferida quando válida, com frequência de bin como fallback.

Todo pico não associado nasce como uma nova trajetória. Uma trajetória sem
observação sobrevive até `max_gap_frames`; depois é encerrada. Lacunas não são
interpoladas. Trajetórias abaixo de `min_track_length` aparecem em
`rejected_tracks`. As métricas expostas são operacionais: duração observada,
contagens de observações e lacunas, deriva e dispersão de frequência, amplitude
e custos de associação.

Em cruzamentos, o método de distância instantânea é determinístico e preserva
as restrições um-para-um, mas pode trocar identidades. Isso é uma limitação
documentada, não uma alegação de continuidade física.

## Validação

Foram adicionados 9 testes, elevando a suíte de 85 para 94 casos. Eles cobrem
senoides estacionárias, duas componentes de amplitude distinta, chirps crescente
e decrescente, senoide amortecida, silêncio, impulso, lacunas curta e longa,
cruzamento determinístico, invariantes, configurações inválidas, integração com
`AnalysisSettings` e um caso moderado de 300 quadros com três trajetórias.

O caso moderado é um teste funcional informal para centenas de quadros e
dezenas de picos; não é benchmark com limite de tempo dependente de máquina.

## Resultado final e limitações

- `python3 -m pytest -q`: **94 passed**;
- `git diff --check`: aprovado;
- `python3 -m compileall -q belllab`: aprovado;
- não há ferramenta de análise estática configurada em `pyproject.toml`.

O próximo módulo recomendado é refinamento da detecção por quadro em sinais
reais e, somente depois, uma associação com previsão curta e validação por
gravações multissensoriais. Uma futura interpretação modal deverá permanecer
uma etapa separada, fundamentada em critérios físicos adicionais.
