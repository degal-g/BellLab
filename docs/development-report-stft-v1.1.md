# Development Report — STFT v1.1

**Data da revisão:** 2026-07-23  
**Estado inicial:** 68 testes aprovados.

## Escopo

Esta revisão encerra a validação inicial da STFT. Não foram implementados
detecção de picos por quadro, tracking, associação temporal, análise modal,
gráficos ou processamento em lote.

## Problemas encontrados e correções

- A documentação do módulo `spectrum.py` ainda dizia que STFT não existia.
  Ela agora descreve FFT estacionária, picos espectrais e STFT, sem antecipar
  tracking ou análise modal.
- Anotações `Literal` não eram validadas em tempo de execução. As configurações
  de FFT, picos e STFT agora rejeitam escolhas fechadas não suportadas.
- A remoção de média da STFT era operacionalmente por quadro, mas descrita de
  forma genérica. Ela é agora registrada como `detrend_method="frame_mean"`,
  ou `"none"` quando desativada.
- A FFT e a STFT tinham a mesma fórmula unilateral em locais distintos. Ambas
  passaram a usar a rotina privada comum de normalização de amplitude.

## Arquivos alterados

- `belllab/config.py`;
- `belllab/spectrum.py`;
- `tests/test_stft.py`;
- `README.md`;
- `docs/RFC-0001-scientific-specification.md`;
- este relatório.

## Contratos e políticas finais

`STFTSettings.channel_policy` aceita somente `"select"` ou `"mean"`.
`"select"` é o padrão e exige um índice válido; `"mean"` só combina canais
por solicitação explícita. Janela, escala, normalização estacionária, métodos
de pico, método de ruído e ordenação também são validados em runtime.

A STFT produz uma matriz unilateral `values[frequency_index, time_index]`.
Os tempos são centros de janela:

`(start_index + frame_index * hop_length + window_length / 2) / sample_rate`.

O intervalo é semiaberto, `[start_time_s, end_time_s)`. Os segundos são
convertidos por teto para índices de amostra, e o resultado registra o
intervalo efetivamente selecionado, derivado desses índices. Sem padding e
para `N >= W`, há `1 + floor((N - W) / H)` quadros. O restante final
incompleto é descartado. Com `pad_end=True`, apenas as amostras necessárias
para completar o último quadro são adicionadas; esse padding não altera a
duração real da gravação.

A normalização é a mesma da FFT estacionária: magnitude unilateral de
amplitude de pico, corrigida pelo ganho coerente da janela, com duplicação dos
bins internos e tratamento distinto de DC e Nyquist. `dbfs` significa
amplitude dBFS com referência linear 1.0; silêncio pode conter `-inf`, nunca
NaN. Zero padding espectral reduz somente o espaçamento da grade de bins, não
a resolução física.

## Diagnósticos auditáveis

Quando ocorrem, a STFT registra seleção de canal ou média explícita, remoção
`frame_mean`, padding final e sua quantidade, descarte de segmento final,
trecho menor que a janela, corte de faixa de frequências, zero padding
espectral e `-inf` em dBFS por silêncio. Os campos estruturados preservam
`padded_samples`, `discarded_samples` e `detrend_method`.

## Validação quantitativa

Foram adicionados 17 testes, elevando a suíte de 68 para 85 casos. Eles cobrem
senoide centrada em bin, chirp, senoide amortecida, impulso, silêncio em dBFS,
offset DC com e sem `frame_mean`, padding, descarte final, sinal curto,
zero padding espectral, recorte de frequência, multicanal em oposição de fase,
opções inválidas em runtime, reprodutibilidade e equivalência FFT × STFT.

No teste de equivalência, um único quadro da STFT e uma FFT estacionária com o
mesmo trecho, janela, `n_fft`, escala e política de média são comparados bin a
bin, incluindo DC e Nyquist, com tolerância absoluta e relativa de `1e-12`.

## Validação final

- `python3 -m pytest -q`: **85 passed**;
- `git diff --check`: aprovado;
- não há ferramenta estática configurada em `pyproject.toml`.

## Limitações conhecidas e próximo passo

Os diagnósticos são deliberadamente simples e não constituem logging nem
proveniência completa. A amplitude e a caracterização temporal ainda devem ser
validadas contra gravações reais, diferentes conversores, ruído ambiental e
janelas de aquisição. A próxima etapa recomendada é detecção de picos por
quadro como API independente, seguida apenas depois por associação temporal;
isso não deve ser confundido com identificação modal física.

Uma futura divisão em `spectrum.py`, `peaks.py` e `time_frequency.py` pode
melhorar coesão, mas foi conscientemente adiada para preservar a API e evitar
uma refatoração ampla.
