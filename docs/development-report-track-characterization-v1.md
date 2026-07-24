# Development Report — Track Characterization v1

**Data:** 2026-07-23  
**Estado inicial:** 94 testes aprovados.

## Mudanças

Foram fortalecidos contratos de picos por quadro e tracking em `types.py`,
`results.py` e `tracking.py`. `TimeFrequencyPeakResults` agora valida índices,
tempos contra a STFT, contagens e unidades coerentes. Unidades lineares e dBFS
de amplitude não podem ser misturadas; capitalização é normalizada apenas para
aliases conhecidos.

`SpectralTrack.gap_count` passou a significar número de intervalos de lacuna.
`total_missing_frames` contém a soma dos quadros internos ausentes e
`largest_gap_frames` a maior lacuna. `tracks_reaching_final_frame` substitui a
semântica ambígua do resultado offline; `active_track_count` é alias legado.

Foi adicionado `SpectralTrackCharacterization` e
`characterize_spectral_track`. A caracterização usa frequência refinada quando
disponível, calcula inclinação/resíduo de frequência, cobertura e métricas de
amplitude. Em amplitude linear positiva, ajusta log-amplitude versus tempo e
reporta `decay_tau_s` somente para inclinação negativa; em dBFS ajusta nível
diretamente. Esses ajustes são operacionais e não estimativas de amortecimento
modal físico.

## Validação e limitações

Foram adicionados 2 testes, elevando a suíte de 94 para **96 passed**. Eles
cobrem a nova semântica de lacunas, tracks que alcançam o último quadro,
unidades incompatíveis e caracterização constante. `pytest`, `compileall` e
`git diff --check` passaram; não há ferramenta estática configurada.

O método Húngaro permanece determinístico e um-para-um. Cruzamentos podem
trocar identidades. O resultado agora registra contagens operacionais de
associações ambíguas e próximas do limiar, além da menor margem observada;
`ambiguity_margin` é um limiar heurístico, não uma probabilidade calibrada.
Previsão linear foi deliberadamente adiada: exige validação adicional.
Nenhuma trajetória ou caracterização é convertida em `ModalMode`.

O próximo passo recomendado é calibrar diagnósticos de ambiguidade e previsão
curta com gravações reais e só então definir critérios para candidatos modais
operacionais, ainda separados de modos físicos validados.
