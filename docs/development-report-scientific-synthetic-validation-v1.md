# Development Report: Scientific Synthetic Validation v1

1. Data: 2026-07-29.

2. Branch: `feature/scientific-synthetic-validation`.

3. Estado inicial: branch confirmada fora de `main`, arvore limpa, `pytest` e
   `pytest -W error` com 1075 testes aprovados antes de qualquer edicao.

4. Arquivos criados:
   - `belllab/synthetic_validation.py`;
   - `tests/test_synthetic_validation.py`;
   - `docs/development-report-scientific-synthetic-validation-v1.md`.

5. Arquivos alterados:
   - `README.md`;
   - `docs/RFC-0001-scientific-specification.md`;
   - `belllab/__init__.py`.

6. Principio cientifico:
   recuperacao correta em sinal sintetico nao garante validade em dados reais;
   erro baixo em um cenario nao prova robustez geral; aprovacao de threshold
   nao e prova fisica; falha de recuperacao nao invalida universalmente o
   metodo; verdade sintetica nao e verdade fisica experimental.

7. Arquitetura da validacao:
   `SyntheticValidationScenario` declara componentes e expectativas;
   `SyntheticGroundTruth` e gerado antes da analise; `SyntheticPipelineOutput`
   registra estagios publicos executados e falhas; validacoes especificas
   comparam verdade conhecida com resultados recuperados; resultados de cenario,
   campanha e Monte Carlo preservam contagens, status, razoes e diagnosticos.

8. Geracao de sinais:
   `SyntheticDampedComponent` suporta frequencia constante, drift linear,
   trajetoria por segmentos, modulacao senoidal, crossing operacional e
   amostras customizadas. Envelopes suportam decaimento exponencial,
   onset tardio, crescimento tardio, recuperacao, envelope por segmentos,
   batimento operacional, amplitude constante e amostras customizadas.

9. Ground truth:
   a verdade e criada antes de qualquer chamada ao pipeline. Ela preserva
   `Signal` limpo, ruido, observado, eixo temporal, frequencias conhecidas,
   tau, Q, bandwidth, presenca de componentes, associacoes, cadeias e pares
   esperados de evidencia operacional de possivel redistribuicao.

10. Pipeline executado:
   `run_synthetic_pipeline` usa APIs publicas: `analyze_temporal`,
   `analyze_spectrum`, `detect_spectral_peaks`, `analyze_stft`,
   `detect_stft_peaks`, `track_spectral_peaks`, `characterize_spectral_track`,
   `select_modal_candidates` e, quando ha envelopes suficientes,
   `evaluate_modal_energy_exchange`. Estagios que exigem entradas nao
   disponiveis sao registrados em diagnosticos, sem continuidade silenciosa.

11. Metricas de frequencia:
   `SyntheticFrequencyValidation` calcula erro absoluto, erro relativo, erro
   assinado e, para trajetorias, RMSE, MAE, erro maximo, erro de slope e erro
   de mudanca total. Limites sao inclusivos.

12. Metricas de tau:
   `SyntheticDecayValidation` compara tau recuperado com tau conhecido por
   erro absoluto, erro relativo e erro logaritmico. Ausencia de tau permanece
   `None`, sem substituicao por zero.

13. Metricas de Q:
   `SyntheticQValidation` usa a mesma convencao sintetica compativel
   `Q = pi * f * tau`. Para o cenario basico, `f = 500 Hz` e `tau = 2.0 s`
   produzem `Q = 3141.592653589793`.

14. Metricas de bandwidth:
   `SyntheticBandwidthValidation` compara largura verdadeira sintetica quando
   identificavel. Para `tau = 2.0 s`, a largura compativel pela relacao
   `Q = f / bandwidth` e `1 / (pi * tau) = 0.15915494309189535 Hz`.
   No cenario basico, a largura operacional recuperada foi
   aproximadamente `0.19122186682500342 Hz`, erro relativo `0.20148`, dentro
   da tolerancia configurada.

15. Tracking:
   `SyntheticTrackingValidation` compara tracks recuperados com componentes
   verdadeiros por uma politica explicita de frequencia mais proxima, usada
   somente na validacao. O tracking do BellLab nao e corrigido pela verdade.

16. Candidatos:
   `SyntheticCandidateValidation` avalia contagem esperada, contagem
   recuperada, candidatos perdidos, falsos candidatos, candidatos esperados
   rejeitados e candidatos aceitos inesperados.

17. Associacoes:
   `SyntheticAssociationValidation` compara pares canonicos por conteudo,
   calcula precisao, recall e F1, e separa emergentes/desaparecidos esperados
   e recuperados.

18. Cadeias:
   `SyntheticChainValidation` compara cadeias por nos e arestas canonicas,
   sem assumir que IDs sinteticos sejam iguais aos IDs operacionais. Cadeias
   vazias esperadas e recuperadas sao um resultado valido.

19. Hipoteses:
   `SyntheticModalHypothesisValidation` compara status esperados e recuperados.
   Aceitacao sintetica nao cria `ModalMode` e nao prova identidade modal
   fisica.

20. Energia operacional:
   `SyntheticEnergyExchangeValidation` compara pares esperados de evidencia
   operacional de possivel redistribuicao com pares suportados pela camada de
   energia. Falsos positivos, falsos negativos, inconclusivos e erros de lag
   permanecem explicitos.

21. Monte Carlo:
   `SyntheticMonteCarloValidation` usa `trial_count`, `trial_seed_stride` e
   seeds deterministicas. O RNG global do NumPy nao e alterado; nao ha selecao
   manual de trials favoraveis.

22. Cenarios implementados:
   `single_ideal`, `multiple_isolated`, `near_modes_resolved`,
   `near_modes_marginal`, `near_modes_unidentifiable`, `beating`,
   `linear_drift`, `frequency_crossing`, `emergence_disappearance`,
   `apparent_split_merge`, `energy_exchange`, `no_energy_exchange`, `noise`,
   `mains_hum`, `clipping`, `short_duration` e `sampling_resolution`.

23. Ruido:
   `SyntheticValidationSettings` suporta ruido branco, ruido rosa operacional,
   SNR alvo, desvio padrao explicito, ruido colorido e hum de rede. Seeds
   iguais reproduzem o mesmo ruido; seeds diferentes alteram somente a
   realizacao observada dependente de ruido.

24. Clipping:
   a camada suporta clipping hard e soft com limiar ou fracao alvo
   configuravel. O clipping observado fica em metadados; regimes severos nao
   sao ocultados.

25. Modos proximos:
   cenarios resolvidos, marginais e nao identificaveis registram interferencia
   de modos vizinhos e limitacao de resolucao. O caso nao identificavel pode
   produzir insuficiencia ou inconclusao correta.

26. Batimento:
   o cenario `beating` usa componentes em 500 Hz e 502 Hz, com periodo
   esperado `1 / |500 - 502| = 0.5 s`. Isso e contexto de possivel batimento,
   nao prova de acoplamento nem redistribuicao fisica.

27. Drift:
   o cenario `linear_drift` usa `f(t) = 500 + 2t`. Para duracao padrao de
   8 s, a frequencia representativa sintetica no meio da janela e 508 Hz.
   O drift e sintetico e nao e classificado como hardening ou softening.

28. Crossing:
   `frequency_crossing` cria duas trajetorias lineares que se cruzam e registra
   risco de track swap. A validacao nao corrige tracking usando a verdade.

29. Emergencia/desaparecimento:
   `emergence_disappearance` inclui componente persistente, componente
   emergente e componente desaparecido, para validar contabilidade parcial sem
   fechamento de lacunas.

30. Split/merge aparente:
   `apparent_split_merge` cria contexto operacional um-para-dois em condicoes
   adjacentes. Nenhum split ou merge fisico e resolvido.

31. Determinismo:
   IDs de cenarios, fingerprints de settings, sinais com mesma seed, campanhas
   e Monte Carlo sao deterministas. A ordem de componentes customizados e
   normalizada por `component_id`.

32. Imutabilidade:
   dataclasses publicas sao congeladas; listas de entrada sao materializadas em
   tuplas; validacao, suavizacao e geracao de ruido usam copias; nao ha cache
   mutavel global.

33. Testes:
   foram adicionados 84 testes cobrindo importacao publica, cenarios
   built-in, ground truth, ruido, clipping, invariantes de settings e
   componentes, limites de frequencia/tau/Q/bandwidth, tracking, candidatos,
   associacoes, cadeias, hipoteses, energia operacional, determinismo,
   imutabilidade, campanha e Monte Carlo.

34. Resultado final:
   a suite cresceu de 1075 para 1159 testes. `pytest` reportou 1159 aprovados.
   A validacao final tambem executa `pytest -W error`,
   `python3 -m compileall -q belllab tests`, `git diff --check` e `git status`.

35. Limitacoes:
   nao ha validacao com gravacoes reais, prova de validade fisica, calibracao
   automatica de thresholds pelo mesmo conjunto, machine learning, modelo
   completo de sino, elementos finitos, osciladores acoplados, identificacao
   causal, resolucao fisica de split/merge, correcao de tracking por ground
   truth, leitura de banco externo, visualizacoes finais, CLI final ou relatorio
   cientifico final de experimento.

36. Proximos passos:
   adicionar campanhas calibradas por familias de instrumentos, fixtures de
   regressao com parametros espectrais variados, validacao externa com dados
   publicos quando houver protocolo aprovado, e relatorios comparativos que
   mantenham separadas validacao sintetica, validacao experimental e
   interpretacao fisica.
