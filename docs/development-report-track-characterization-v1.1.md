# Development Report — Track Characterization v1.1

**Data:** 2026-07-23  
**Estado inicial:** 96 testes aprovados.

## Correções

`SpectralTrack` agora preserva `amplitude_unit` canônica:
`linear_amplitude` ou `dbfs_amplitude`. A unidade é herdada dos picos e nunca
é deduzida por sinais numéricos. Misturas linear/dBFS são rejeitadas pelo
contrato de resultados por quadro.

`characterize_spectral_track` usa exclusivamente essa unidade. Para amplitude
linear positiva, ajusta `ln(A) = b + m t` e calcula `tau = -1/m` apenas para
`m < 0`. Para dBFS, ajusta `L = b + m t` em dB/s e calcula
`tau = -20 / (m ln(10))`, também apenas para inclinação negativa. Nenhum valor
em dB é exponenciado. Pontos inválidos são contabilizados e resultados
indisponíveis são `None` com diagnóstico.

`SpectralTrackCharacterization` passou a registrar unidade, intercepto e
contagens de pontos usados/descartados. As lacunas distinguem intervalos,
quadros ausentes totais e maior lacuna. O resultado offline usa
`tracks_reaching_final_frame`; `active_track_count` é alias legado.

## Associação

O tracking continua Húngaro, determinístico e um-para-um. Agora registra
contagens operacionais de ambiguidade e proximidade ao limiar, além da menor
margem observada. A margem é heurística e não probabilidade. Distâncias são
explicitamente Hz, relativa simétrica (`abs(f2-f1)/max(abs(f1),abs(f2))`) ou
cents. Cruzamentos continuam potencialmente ambíguos.

## Validação

Foram adicionados 4 testes, elevando a suíte de 96 para **100 passed**. Eles
recuperam `tau = 0.2, 1.0, 3.0 s` em amplitude linear com tolerância numérica,
e recuperam `tau = 1 s` em dBFS para inclinação
`-20/ln(10) = -8.685889638 dB/s`. Também verificam unidade incompatível e a
semântica de lacunas/trajetórias finais.

`python3 -m pytest -q`, `python3 -m compileall -q belllab` e
`git diff --check` passaram. Não há análise estática configurada.

Nenhum candidato modal ou `ModalMode` foi criado. A próxima etapa recomendada
é ampliar testes dedicados de margens por linha/coluna e pesos de amplitude com
gravações reais, antes de qualquer critério modal operacional.
