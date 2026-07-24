# Validation Report — Track Amplitude Fit v1

**Estado inicial real:** 114 testes aprovados.  
**Resultado final:** 117 testes aprovados.

Foram adicionados testes de amplitude constante linear e dBFS, crescimento e
filtragem de NaN/+inf. A validação revelou que uma regressão em dBFS constante
podia produzir inclinação negativa residual de ponto flutuante. A implementação
agora usa tolerância explícita de `1e-12`: inclinações com módulo menor ou igual
a esse valor recebem diagnóstico `constant_amplitude` e não produzem tau.

Trajetórias crescentes recebem `amplitude_increasing`, também sem tau. Pares
não finitos são descartados antes de ajuste; o caso testado preservou três
pontos finitos e descartou dois, sem RMSE não finito. Os aliases legados da
caracterização continuam encaminhando a `TrackAmplitudeFit`.

`pytest`, `compileall` e `git diff --check` passaram. Tau continua operacional,
não é Q nem amortecimento modal validado. A próxima rodada recomendada é
ampliar sistematicamente séries ruidosas, dados insuficientes e métricas de
frequência/amplitude sem introduzir candidatos modais.
