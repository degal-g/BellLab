# BellLab — Architecture Review v1

**Data da revisão:** 2026-07-23  
**Escopo auditado:** todo o conteúdo versionável do projeto, incluindo pacote
`belllab`, testes, configuração de empacotamento, README e RFC-0001.  
**Verificação executada:** `python3 -m pytest -q` — **32 testes aprovados**.  
**Natureza da revisão:** diagnóstico técnico e científico; nenhuma alteração de
implementação foi realizada como parte desta revisão.

## Sumário executivo

O BellLab tem uma fundação conceitual sólida: separa modelos de domínio,
resultados, leitura WAV, análise temporal, síntese para testes e futuras áreas
espectrais/modais. A RFC estabelece uma direção científica adequada e a suíte
de testes sintéticos já torna o desenvolvimento do módulo temporal verificável.

Ainda não é uma arquitetura pronta para sustentar, sem revisão de contratos,
uma sequência ampla de algoritmos científicos e resultados publicáveis. O ponto
central é a divergência entre o modelo proposto pela RFC e os objetos que de
fato carregam resultados: `TemporalResults` não contém impacto nem métricas
temporais, `AnalysisSettings` não registra parâmetros, e há duplicação entre
`Recording` e `ProcessingContext`. Essas lacunas afetam diretamente
proveniência, reprodutibilidade e comparação em escala.

## 1. Arquitetura

| Aspecto | Classificação | Justificativa técnica |
| --- | --- | --- |
| Modularidade | **Bom** | O código está dividido de modo compreensível entre `types`, `results`, `recording`, `io`, `temporal`, `synthetic` e módulos futuros. A divisão evita um módulo monolítico. |
| Acoplamento | **Aceitável** | `Recording` referencia resultados individuais e também um `ProcessingContext`, que volta a conter sinal, configurações e os mesmos resultados. Essa dupla representação cria acoplamento e risco de estados divergentes. |
| Coesão | **Bom** | `io` concentra leitura WAV e `synthetic` concentra geração. `temporal` contém análise temporal e seus relatórios, o que é coerente. A função de conveniência `analyze_temporal` mistura uma API agregada incompleta com APIs detalhadas paralelas. |
| Organização dos pacotes | **Bom** | O pacote `instruments` estabelece uma fronteira correta para especializações; os módulos vazios de espectro, modal, gráfico e relatório deixam o roadmap explícito. Ainda faltam fronteiras para campanhas, proveniência e validação. |
| Separação entre domínio científico e infraestrutura | **Bom** | `Signal`, resultados e modelos de domínio não dependem de `soundfile`; a dependência de arquivo está isolada em `io`. Contudo, `Recording.path` e `Signal.path` duplicam infraestrutura/proveniência. |
| Extensibilidade | **Aceitável** | A extensão por famílias instrumentais é conceitualmente correta, mas não há protocolo, classe-base ou contrato de especialização. O modelo de `Experiment` é estritamente binário, não representa campanhas ou conjuntos. |
| Clareza da API pública | **Aceitável** | `belllab.__init__` exporta os principais contratos, mas não expõe os relatórios e funções temporais nem os geradores sintéticos. O retorno de `load_wav` é uma tupla posicional e coexistem APIs diretas e agregadas sem uma convenção única. |

### Observações arquiteturais

- `Recording` continua exigindo um campo chamado `bell_id`, apesar da
  generalização para idiofones. A documentação o chama corretamente de legado,
  porém o contrato público ainda impõe uma semântica específica de sino.
- `Experiment` é, na prática, uma comparação entre duas gravações. O nome
  sugere um experimento mais amplo que deveria poder conter várias gravações,
  condições e execuções.
- Os aliases `BellRecording` e `BellComparison` preservam compatibilidade, mas
  não possuem cronograma, aviso de descontinuação ou política de versão.

## 2. Modelo de dados

| Objeto | Avaliação |
| --- | --- |
| `Signal` | **Aceitável.** Expõe amostras, eixo temporal, duração, canais, unidade e proveniência potencial. Porém `time`, `duration`, `channels` e a forma de `samples` podem discordar sem validação. `path`, `filename`, `sha256` e `loaded_at` não são preenchidos pelo carregador WAV. A representação por tuplas de escalares tem alto custo de memória e conversão para sinais longos. |
| `Recording` | **Aceitável.** Reúne sinal, métricas e resultados, mas duplica o local de armazenamento de resultados com `ProcessingContext`; também duplica caminho de origem com `Signal`. `metadata: Mapping[str, str]` é estreito para metadados científicos estruturados e não garante imutabilidade profunda. |
| `Experiment` | **Precisa revisão.** O contrato exige exatamente `reference` e `candidate`; portanto é uma comparação binária, não um experimento científico reprodutível. Não registra campanha, condições, método, participantes múltiplos ou resultados de comparação. |
| `RecordingMetrics` | **Aceitável.** Os campos básicos de aquisição são adequados. Há redundância entre `max_level_dbfs` e `peak_dbfs`; `peak_dbfs`, `rms_dbfs`, `crest_factor_db`, `clipping_fraction` e `clipping_sample_count` não são preenchidos por `load_wav`. A mistura de campos opcionais e obrigatórios não indica se `None` significa não aplicável, não calculado ou falha. |
| `Envelope` | **Bom.** Método, unidade e parâmetros dão uma boa base de proveniência. Entretanto, não há garantia de que `times_s` e `amplitudes` tenham o mesmo tamanho; um `Mapping` recebido do chamador pode ser mutável, apesar da dataclass congelada. |
| `Spectrum` | **Aceitável.** O contexto de janela, FFT, sobreposição e instante é útil. `overlap` não declara se é fração, amostras ou segundos; não há fase, convenção de normalização ou referência explícita de magnitude. É aceitável como contrato inicial, mas insuficiente para uma publicação. |
| `ModalMode` | **Aceitável.** Frequência, amplitude, amortecimento e Q são um núcleo plausível. Faltam unidade e referência da amplitude, incerteza, método de identificação, intervalo temporal, confiança e identificador estável de rastreamento. |
| `ProcessingContext` | **Precisa revisão.** A intenção é boa, mas replica em outro objeto o que `Recording` já guarda. Não tem identidade de execução, versão de software, hash de entrada, parâmetros efetivos ou ordem das transformações. |
| Results | **Precisa revisão.** `TemporalResults` contém ruído, envelope e ajuste, mas não contém `ImpactReport` nem `TemporalMetrics`, ambos produzidos em `temporal.py`. `NoiseReport` e `NoiseMetrics` duplicam o mesmo conceito com campos/nomenclatura distintos. `SpectrumResults` e `ModalResults` são adequados como cascas iniciais. |
| `AnalysisSettings` | **Precisa revisão.** Uma configuração vazia não centraliza nem preserva parâmetros. A implementação atual de `analyze_temporal` descarta explicitamente o objeto recebido, contrariando a promessa de reprodutibilidade. |

### Ambiguidades científicas relevantes

- A unidade `"digital"` do `Signal` carregado é diferente da escala
  normalizada usada nas métricas em dBFS. A relação é descrita no código, mas
  não é representada pelo tipo de dados.
- `total_energy` temporal é corretamente documentada como medida dependente da
  unidade, mas o nome pode ser interpretado como energia física sem uma
  calibração de impedância/pressão.
- `confidence` em `ImpactReport` e `NoiseReport` não declara significado
  estatístico, método de calibração ou incerteza associada.

## 3. Pipeline científico

### Pipeline atualmente implementado

```text
WAV ou gerador sintético
        ↓
Signal (+ RecordingMetrics no caso WAV)
        ↓
funções temporais independentes
  ├─ ImpactReport
  ├─ NoiseReport
  ├─ TemporalMetrics
  └─ Envelope
        ↓
TemporalResults parcial (somente via analyze_temporal)
        ↓
Spectrum / Modal / Comparison / Report: interfaces não implementadas
```

O fluxo está parcialmente consistente: `Signal` é a entrada comum e as funções
temporais não escrevem arquivos nem geram gráficos. O pipeline real, porém, não
segue integralmente o pipeline da RFC. Não há pré-processamento, persistência de
proveniência, agregação completa de resultados temporais, análise espectral,
modal, comparação ou relatório.

### Gargalos futuros

1. **Agregação de resultados:** resultados diretos não entram integralmente em
   `TemporalResults` nem em `Recording` de forma canônica.
2. **Parâmetros e proveniência:** ausência de configurações efetivas impede
   reproduzir uma execução a partir de um contexto persistido.
3. **Política multicanal:** `temporal.py` usa média dos canais para várias
   métricas; microfones com fase, ganho ou posição distintos podem se cancelar.
4. **Pré-processamento e comparabilidade:** não existem regras para seleção de
   canal, alinhamento, normalização, calibração, recorte ou taxas de amostragem
   heterogêneas.
5. **Escala de armazenamento:** todo sinal é carregado na memória e convertido
   para tuplas Python antes de avançar no pipeline.

## 4. Testabilidade

**Classificação geral: Bom.** Há testes unitários separados por área (`io`,
modelos, tipos, temporal e síntese), e os geradores artificiais permitem criar
oráculos controlados para sinais simples. A suíte é rápida e atualmente passa
integralmente.

### Cobertura conceitual atual

- Leitura de WAV mono PCM16, estéreo float32/float64 e erros de entrada.
- Contratos de modelos, aliases de compatibilidade e campos de proveniência.
- Impacto isolado, ruído com cauda simples, métricas de sinal constante,
  silêncio e envelopes temporais básicos.
- Senoides, soma, amortecimento, impulso, ruído determinístico, clipping,
  offset DC e mistura.

### Lacunas para validação científica robusta

- Fixtures de referência independentes (gravações calibradas ou dados publicados)
  e valores esperados rastreáveis.
- Testes de PCM U8, PCM24, PCM32, WAV multicanal maior que dois canais, arquivo
  vazio, sinal muito longo, amplitudes fora de fundo de escala e metadados de
  origem do WAV.
- Testes multicanal com defasagem e cancelamento, múltiplos impactos, ruído não
  estacionário, clipping parcial, offset DC alto e sinais com baixa SNR.
- Testes de propriedades, limites numéricos, invariantes de dimensionalidade e
  regressões contra versões anteriores.
- Validação quantitativa da inclinação espectral do ruído rosa e da precisão dos
  envelopes/impacto com conjuntos sintéticos parametrizados.
- Integração contínua, relatório de cobertura, lint, formatação, análise de
  tipos e matrizes de Python/dependências não estão configurados.

## 5. Escalabilidade

| Cenário | Avaliação | Motivo |
| --- | --- | --- |
| Dezenas de gravações | **Bom** | Para gravações curtas, a API em memória e os testes atuais são suficientes para processamento manual ou em scripts. |
| Centenas de gravações | **Aceitável** | Falta catálogo de metadados, execução em lote, persistência de resultados, paralelismo e política de falhas. A conversão para tuplas amplia consumo de memória. |
| Milhares de gravações | **Precisa revisão** | Não há carregamento preguiçoso, armazenamento externo, índice de experimentos, cache, particionamento, monitoramento ou formato de resultados serializável. |
| Instrumentos diferentes | **Aceitável** | A independência do núcleo e `instruments/` são boas bases, mas ainda não há contratos concretos de especialização. |
| Campanhas experimentais | **Precisa revisão** | `Experiment` só compara dois `Recording`; não representa campanha, condições, réplica, sessão ou conjunto de medições. |
| Múltiplos microfones | **Aceitável** | `Signal` e `io` preservam canais, porém não há identidade/posição/calibração de canal e análises temporais combinam canais por média. |
| Taxas de amostragem distintas | **Aceitável** | Cada `Signal` preserva sua taxa. Não existe, porém, política de comparação, reamostragem, validação de Nyquist ou normalização temporal. |

## 6. Documentação

| Artefato | Classificação | Avaliação |
| --- | --- | --- |
| README | **Bom** | Comunica escopo generalizado, filosofia e estrutura. É conciso demais para uma API científica: não tem exemplo de carregamento, exemplo temporal, convenções de retorno ou política de versões. |
| RFC-0001 | **Bom** | Define objetivos, pipeline, grandezas, convenções e extensibilidade com boa linguagem científica. Está mais madura que a implementação e não define critérios operacionais de incerteza, versões de método ou qualidade de dados. |
| BSS | **Precisa revisão** | Não há documento BSS identificado no repositório. Se a sigla se refere a uma especificação complementar, ela precisa ser criada, nomeada e vinculada à RFC. |
| Docstrings | **Bom** | A maioria das APIs públicas documenta entradas, saídas e limitações; `temporal.py` também cita referências. Algumas docstrings históricas dizem que análises futuras não existem, embora o módulo temporal já exista. |
| Tipagem | **Bom** | Type hints modernos e dataclasses são usados consistentemente. Ainda faltam tipos para unidades, métodos, valores de configuração e resultados serializáveis. |
| Documentação pública | **Aceitável** | Não há referência gerada da API, guia de contribuição, política de depreciação, changelog, exemplos executáveis ou documentação de reprodutibilidade operacional. |

Há também uma inconsistência de posicionamento: `README.md` descreve um framework
de idiofones, enquanto `pyproject.toml` ainda se apresenta como toolkit para
sinos históricos.

## 7. Engenharia de software

### Pontos positivos

- Dependências pequenas e apropriadas ao estágio atual (`numpy`, `scipy`,
  `soundfile`, `pytest`).
- Licença MIT, empacotamento por `pyproject.toml` e testes isolados por módulo.
- Exceções específicas para erros de formato e leitura WAV.
- Uso de dataclasses congeladas para a maior parte dos resultados.

### Code smells e riscos de manutenção

1. **Contratos redundantes:** `Recording` e `ProcessingContext` podem conter o
   mesmo sinal e resultados; `RecordingMetrics` possui duas representações
   potenciais de pico em dBFS.
2. **Imutabilidade superficial:** `Envelope.parameters` e `Recording.metadata`
   aceitam mappings que podem ser alterados externamente após a construção.
3. **Invariantes não validados:** `Signal` aceita comprimento incompatível entre
   canais, tempo, duração e `channels`; `Envelope` e `Spectrum` não validam
   pares de séries nem ordenação de eixos.
4. **Generalização incompleta:** `bell_id` permanece na classe central e
   `Experiment` tem semântica de comparação binária.
5. **Implementação temporal específica:** `detect_impact` calcula a janela basal
   usando `samples.shape[0]` (quantidade de canais), e não o número de amostras;
   em um sinal mono isso reduz a base a uma amostra. É um risco funcional para a
   confiança e o limiar adaptativo.
6. **API agregada incompleta:** `analyze_temporal` ignora `AnalysisSettings`,
   retorna somente uma fração dos resultados disponíveis e embute uma definição
   de SNR sem registrar o método no resultado.
7. **Desempenho previsível:** `envelope_moving_peak` percorre janelas em Python,
   com custo crescente para gravações longas; `Signal` converte arrays em
   tuplas de objetos Python.
8. **Dependências e lançamento:** limites inferiores amplos sem lockfile são
   adequados para biblioteca, mas não bastam para reprodução estrita. A lista
   em `requirements.txt` duplica as dependências do `pyproject.toml`.

## 8. Ciência, reprodutibilidade e auditoria

**Classificação: Aceitável.** A intenção científica é forte e explícita na RFC.
Métodos de envelope registram nomes e parâmetros; `ImpactReport` identifica o
método; os testes sintéticos tornam parte do comportamento auditável. Isso é
uma base útil para pesquisa exploratória.

Para pesquisa reprodutível, publicação e auditoria formal, ainda faltam:

- hash e proveniência efetivamente preenchidos para arquivos de origem;
- versão do pacote, versão de método e dependências anexadas a cada resultado;
- parâmetros efetivos em `AnalysisSettings` e nos relatórios;
- unidade, referência e calibração sistemáticas para toda grandeza;
- incertezas, condições de validade, diagnósticos e qualidade de ajuste;
- serialização estável de resultados, configuração e ambiente de execução;
- referências de validação independentes e política de dados experimentais;
- distinção formal entre valores físicos calibrados e quantidades digitais
  normalizadas.

O desenho atual permite evoluir para esses requisitos, mas não os torna
obrigatórios nem os preserva automaticamente.

## 9. Roadmap recomendado

1. **Fechar os contratos de dados e proveniência antes de novos algoritmos.**
   Definir uma única posse de sinal/resultados, o papel de `Experiment`,
   invariantes de `Signal` e o significado de campos opcionais. Acrescentar
   identidade de execução, versões e parâmetros efetivos.
2. **Definir `AnalysisSettings` por domínio e alinhar Results.** Cada análise
   deve receber configuração tipada e devolver um resultado completo,
   incluindo método, parâmetros, unidades, qualidade e incerteza quando
   aplicável. Integrar `ImpactReport` e `TemporalMetrics` ao resultado temporal.
3. **Fortalecer I/O e pré-processamento.** Preencher proveniência do arquivo,
   validar formatos/canais, documentar normalização e criar contratos para
   seleção de canal, recorte, calibração e reamostragem.
4. **Validar e consolidar a análise temporal existente.** Corrigir os riscos de
   base de impacto, formalizar SNR/faixa dinâmica/confiança, testar cenários
   multicanal e de baixa SNR, e comparar com referências independentes.
5. **Implementar análise espectral.** Após convenções de amplitude, janelas e
   taxa de amostragem estarem fixadas, produzir `SpectrumResults` completos.
6. **Implementar análise tempo-frequência e modal.** STFT, rastreamento e
   identificação modal devem depender dos contratos espectrais estabilizados e
   registrar incertezas e critérios de qualidade.
7. **Redesenhar comparação/campanhas.** Separar entidade de experimento,
   campanha e comparação, então implementar métricas compatíveis entre taxas,
   canais e condições de aquisição.
8. **Relatórios, exportação e visualização.** Só depois que resultados forem
   serializáveis e auditáveis, gerar artefatos para pesquisa e publicação.
9. **Escala e governança.** Adicionar execução em lote, catálogo de metadados,
   cache, CI, cobertura, lint, type checking, changelog e política formal de
   depreciação.

## 10. Conclusão

### Pontos fortes

- Visão científica clara, documentada na RFC e aplicável a idiofones diversos.
- Modularidade inicial saudável e baixo acoplamento de dependências externas.
- Dados tipados e docstrings amplas para o estágio do projeto.
- Primeira análise temporal isolada de I/O e apresentação.
- Geradores sintéticos úteis e testes rápidos, determinísticos e verdes.
- Estratégia explícita de compatibilidade para a transição de sinos para
  idiofones percutidos.

### Fragilidades

- Contratos de resultados temporais não representam tudo o que o módulo produz.
- `AnalysisSettings` é vazio e não participa da execução temporal atual.
- Redundâncias entre `Recording`, `ProcessingContext` e métricas.
- Generalização semântica incompleta por meio do campo obrigatório `bell_id`.
- Documentação de API e metadados de proveniência ainda não são operacionais.

### Riscos

- Acumular algoritmos sobre contratos de resultados incompletos, exigindo uma
  migração difícil depois que usuários e dados existirem.
- Produzir valores aparentemente científicos sem unidade, configuração,
  incerteza ou método suficientes para reprodução.
- Obter resultados incorretos para múltiplos microfones por combinação implícita
  de canais ou para impactos reais por limiares heurísticos não validados.
- Encontrar limites de memória e organização ao passar de scripts individuais
  para campanhas com centenas ou milhares de arquivos.

### Recomendações prioritárias

1. Estabilizar modelos de proveniência, configurações e resultados antes de
   implementar espectro e modos.
2. Definir uma política científica para unidades, calibração, multicanal e
   dados ausentes.
3. Transformar os testes sintéticos em uma bateria parametrizada com referências
   independentes e cenários adversariais.
4. Formalizar `Experiment` e campanhas como modelos distintos da comparação
   binária.
5. Implantar CI, lint, type checking e cobertura antes de aumentar a superfície
   de algoritmos.

### Arquitetura pronta para iniciar os algoritmos?

**NÃO — não para iniciar a próxima camada ampla de algoritmos científicos.**

O BellLab já pode continuar a validação e a consolidação do módulo temporal em
escopo controlado. Entretanto, implementar FFT/STFT, identificação modal e
comparações sobre os contratos atuais consolidaria ambiguidades de resultado,
configuração e proveniência. A revisão dos contratos descrita nos dois primeiros
itens do roadmap deve anteceder a expansão algorítmica.

### Notas

| Dimensão | Nota (0–10) | Fundamentação resumida |
| --- | ---: | --- |
| Arquitetura | **6,5** | Boa modularização inicial; contratos agregados e semântica de experimento exigem revisão. |
| Modelo de dados | **5,5** | Tipagem e objetos científicos promissores, com redundâncias, invariantes ausentes e resultados desalinhados. |
| Escalabilidade | **4,0** | Adequado para poucos sinais em memória; sem infraestrutura para campanhas grandes ou processamento em lote. |
| Documentação | **7,0** | RFC e docstrings fortes; README/API operacional, BSS, exemplos e governança ainda insuficientes. |
| Testabilidade | **7,0** | Boa base unitária e sintética; faltam validações independentes, bordas científicas e automação de qualidade. |
| Preparação para pesquisa científica | **5,0** | Objetivo e linguagem científica corretos, mas proveniência, parâmetros, incerteza e validação externa não estão completos. |
| **BellLab v1.0 — nota geral** | **5,8** | Fundação promissora de pré-alfa; requer estabilização de contratos e reprodutibilidade antes de uma expansão científica ampla. |
