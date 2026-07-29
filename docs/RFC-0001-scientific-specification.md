# RFC-0001 — Scientific Specification

**Status:** Referência científica inicial  
**Projeto:** BellLab  
**Idioma:** Português

Este documento estabelece a especificação funcional e científica de referência
do BellLab. As futuras implementações devem preservar seus princípios,
convenções e requisitos de reprodutibilidade. Esta RFC define *o que* o sistema
deve representar, medir e comunicar; ela não prescreve algoritmos, fórmulas ou
detalhes de implementação.

## 1. Objetivo

O BellLab é um framework científico de código aberto para análise temporal,
espectral e modal de idiofones percutidos a partir de gravações no formato WAV.
Seu propósito é organizar, analisar, comparar e comunicar evidências acústicas
de forma consistente, rastreável e apropriada para pesquisa.

O objetivo principal é produzir resultados reproduzíveis para estudos em
acústica, patrimônio histórico, musicologia, conservação e engenharia acústica.
Sinos históricos constituem o primeiro domínio de aplicação, sem limitar o
núcleo a essa família instrumental.
O software deve permitir que uma gravação, seus metadados, as escolhas de
análise e seus resultados sejam preservados como um conjunto científico
auditável.

O BellLab não substitui a interpretação especializada. Ele fornece uma base
computacional explícita para que hipóteses e conclusões possam ser examinadas,
comparadas e reproduzidas por outros pesquisadores.

### Supported Instrument Families

O núcleo científico é aplicável, entre outras, às seguintes famílias e classes
de instrumentos ou objetos sonoros percutidos:

- Bells;
- Carillons;
- Gongs;
- Singing bowls;
- Cymbals;
- Metallic plates;
- Rectangular plates;
- Circular plates;
- Lithophones;
- Wooden idiophones;
- Ceramic vessels;
- Archaeological sounding objects.

Especializações futuras para essas famílias poderão acrescentar vocabulário,
metadados e convenções de domínio sem alterar o núcleo do BellLab.

---

## 2. Escopo

O BellLab deverá oferecer estruturas e análises capazes de descrever, quando os
dados e os parâmetros disponíveis permitirem, os seguintes aspectos de uma
gravação e de um idiofone percutido:

- duração do sinal;
- detecção do impacto ou início do evento sonoro;
- caracterização de ruído;
- relação sinal-ruído (SNR);
- faixa dinâmica;
- envelope temporal;
- tempos característicos de decaimento;
- representação por FFT;
- representação tempo-frequência por STFT, com convenções explícitas de janela,
  avanço temporal, escala, canais e padding;
- detecção de picos por quadro e associação em trajetórias espectrais como
  observações matemáticas, sem lhes atribuir automaticamente significado modal;
- caracterização descritiva dessas trajetórias por frequência, amplitude,
  regressões operacionais, cobertura e lacunas, também sem promoção modal;
- caracterização espectral global por potência linear, incluindo distribuição,
  rolloff, flatness, entropia, densidade e largura de picos, frações tonais e
  energia por bandas, sem classificar automaticamente regimes físicos;
- caracterização espectral resolvida no tempo por quadros após o impacto,
  reutilizando métricas globais em potência linear para descrever energia,
  centroide, rolloff, flatness, entropia, densidade, frações tonais,
  ocupação, bandas, tendências e pontos de mudança operacionais, sem
  classificar transições de regime;
- comparação descritiva entre condições dinâmicas nominais (`pp`, `p`, `mf`,
  `f`, `ff`) a partir de resumos por condição, agregando repetições e
  comparando métricas de excitação, espectrais globais e espectrais resolvidas
  no tempo sob critérios explícitos de comparabilidade instrumental e
  espectral, sem classificar linearidade, provar não linearidade ou associar
  candidatos individuais entre condições;
- associação operacional, conservadora e auditável de candidatos modais
  individuais entre condições dinâmicas nominalmente adjacentes (`pp -> p`,
  `p -> mf`, `mf -> f`, `f -> ff`), usando candidatos já caracterizados e
  critérios configuráveis de frequência, tau, tracking e evidência
  pré-impacto. A correspondência resultante não estabelece identidade modal
  física, modo preservado, prova de linearidade ou prova de não linearidade;
- encadeamento determinístico e auditável dessas correspondências operacionais
  adjacentes ao longo de subsequências contíguas da ordem nominal `pp -> p ->
  mf -> f -> ff`, sem recalcular custos, criar associações não adjacentes,
  fechar lacunas, resolver split/merge ou promover cadeias a identidade modal
  física, família modal ou `ModalMode`;
- avaliação conservadora, explícita e auditável de cadeias operacionais como
  `ModalHypothesis`, produzindo estados como aceita, aceita com ressalvas,
  inconclusiva, rejeitada, evidência insuficiente ou entrada inválida. Uma
  hipótese modal operacional aceita significa apenas que a cadeia satisfez
  critérios configuráveis de cobertura, continuidade frequencial, qualidade de
  associação, tracking, tau, evidência pré-impacto e contexto estrutural; ela
  não comprova modo físico, identidade modal definitiva, linearidade,
  não linearidade, divisão física ou fusão física;
- estimação operacional, conservadora e auditável de parâmetros associados a
  `ModalHypothesis`, usando somente valores já presentes na hipótese, nos
  candidatos caracterizados, matches aceitos, cadeias construídas e
  diagnósticos existentes. Frequência representativa, trajetória, drift, tau,
  taxa de decaimento e incertezas operacionais são sínteses quantitativas
  configuráveis, não frequências modais exatas, constantes físicas invariáveis,
  prova de identidade modal, prova de linearidade ou prova de não linearidade;
- estimativas operacionais de fator de qualidade `Q` e largura de banda
  associadas a `ModalParameterEstimate`, derivadas de convenções matemáticas
  explícitas, parâmetros modais operacionais e espectros ou larguras de pico já
  calculados, sem transformar hipótese modal em modo físico comprovado;
- validação sintética científica controlada, gerando cenários com verdade
  conhecida por construção e comparando essa verdade com resultados
  recuperados por APIs públicas do BellLab. Essa validação testa comportamento
  operacional em cenários controlados; ela não prova validade geral em
  gravações reais, não usa a verdade sintética como entrada oculta dos
  estimadores e permite que cenários não identificáveis terminem corretamente
  como insuficientes ou inconclusivos;
- descritores operacionais de caráter espectral observado, calculados
  exclusivamente a partir de métricas já existentes e organizados por dimensões
  independentes como estrutura espectral, evolução temporal, preservação
  operacional de linhas e confiabilidade da evidência, com critérios, pesos,
  scores, conflitos e limitações auditáveis, sem constituir prova de
  linearidade, não linearidade, caos, transição física de regime ou modo físico;
- seleção reversível de trajetórias caracterizadas como candidatos modais
  operacionais por critérios configuráveis e auditáveis, sem identificá-las
  automaticamente como modos físicos;
- evidência operacional pré/pós-impacto para distinguir componentes emergentes,
  amplificadas, persistentes e reexcitadas sem rejeitar automaticamente linhas
  já presentes antes do impacto;
- visualização do tipo waterfall;
- identificação modal;
- rastreamento modal ao longo do tempo;
- energia modal;
- comparação entre gravações;
- geração automática de relatórios científicos.

O escopo inclui a associação desses resultados a metadados de aquisição,
identificação do objeto sonoro e contexto experimental. Não pressupõe que toda gravação
permita estimar todas as grandezas: a disponibilidade e a qualidade de cada
resultado devem ser informadas de maneira explícita.

---

## 3. Pipeline científico

O BellLab adota o seguinte pipeline conceitual. Cada etapa deve preservar a
rastreabilidade das entradas, configurações e saídas que lhe dizem respeito.

```text
Aquisição
    ↓
Leitura
    ↓
Pré-processamento
    ↓
Análise temporal
    ↓
Análise espectral
    ↓
Análise modal
    ↓
Comparação
    ↓
Relatório
```

- **Aquisição:** registra o contexto em que a gravação foi realizada, incluindo
  informações disponíveis sobre o idiofone, o ambiente e a instrumentação.
- **Leitura:** incorpora o arquivo WAV e suas propriedades observáveis ao
  contexto científico do experimento.
- **Pré-processamento:** prepara uma representação do sinal para as análises,
  sempre com parâmetros e transformações documentados.
- **Análise temporal:** descreve a evolução do sinal no tempo, incluindo impacto,
  amplitude, ruído e decaimento.
- **Análise espectral:** descreve a distribuição temporal ou global do conteúdo
  acústico em frequência.
- **Análise modal:** organiza componentes modais identificáveis e suas
  propriedades acústicas ao longo do tempo.
- **Comparação:** relaciona resultados de duas ou mais gravações sob critérios
  explícitos e compatíveis.
- **Relatório:** reúne dados de entrada, configurações, resultados, limitações e
  proveniência em uma apresentação adequada à comunicação científica.

As etapas podem ser estendidas ou parcialmente executadas, mas uma saída nunca
deve ocultar quais etapas contribuíram para sua produção.

---

## 4. Objetos científicos

### Signal

`Signal` representa o sinal de áudio carregado em memória. Cientificamente, é a
observação digital de uma emissão sonora: preserva amostras, referência temporal,
taxa de amostragem, duração, canais e unidade de amplitude. Ele é a entrada
direta das análises.

### Recording

`Recording` representa a gravação como objeto de pesquisa. Ela associa um
`Signal` à identidade do idiofone, à origem do arquivo, aos metadados de
aquisição e aos resultados derivados. Uma gravação deve permanecer distinguível
de suas análises: diferentes configurações podem produzir diferentes resultados
para a mesma observação original.

### Envelope

`Envelope` representa a evolução temporal da amplitude característica do sinal.
É um objeto de apoio para descrever o ataque, a sustentação, o ruído residual e
o decaimento acústico de uma emissão percutida.

### Spectrum

`Spectrum` representa o conteúdo acústico em função da frequência para um
contexto de análise declarado. Ele permite descrever componentes, bandas e
relações espectrais sem confundir a representação com uma interpretação modal.

### GlobalSpectralCharacterization

`GlobalSpectralCharacterization` descreve o espectro de uma gravação como uma
distribuição global de potência linear. Preserva domínio original, conversão,
janela, detrending, faixa, grade FFT e resolução física limitada pela duração,
além de momentos, rolloffs, flatness, entropia, crest factor espectral, picos,
larguras, densidade, espaçamento, energia tonal operacional, resíduo, ocupação
e bandas configuráveis. Essas métricas não constituem diagnóstico definitivo
de ruído, caos, turbulência, não linearidade, acoplamento ou identidade modal.
Picos globais permanecem observações matemáticas e não são promovidos a
`ModalCandidate` ou `ModalMode`.

### TimeResolvedSpectralCharacterization

`TimeResolvedSpectralCharacterization` descreve a evolução temporal de métricas
espectrais globais calculadas quadro a quadro depois de um impacto. A política
de quadros, duração, hop, padding temporal, janela, detrending, FFT, faixa de
frequência, detector de picos, bandas, regiões temporais, regressões e pontos
de mudança deve ser explícita e auditável. Quadros silenciosos, fracos ou não
finitos permanecem representados como inválidos, com motivo estruturado. O
resultado pode relatar ataque de banda larga, cauda mais tonal, deslocamento de
centroide, mudanças de densidade e persistência de bandas, mas não prova
transição física de regime, não linearidade, caos nem identidade modal.

### DynamicConditionComparisonResult

`DynamicConditionComparisonResult` descreve como métricas já calculadas mudam ao
longo da ordem dinâmica nominal `pp < p < mf < f < ff`. A unidade de comparação
é o resumo de cada condição, não uma única gravação isolada: repetições da mesma
condição são agregadas por estatísticas descritivas robustas e auditáveis. O
resultado preserva condições ausentes, incompatibilidades instrumentais,
incompatibilidades espectrais, efeitos de clipping, pares adjacentes ou saltos
nominais, comparações contra uma referência configurável e sequências por
métrica com monotonicidade operacional. A ordem dos rótulos musicais não garante
ordenação física de intensidade, e nenhuma diferença observada constitui por si
só prova de linearidade, não linearidade, mudança modal ou transição de regime.

### ResponseRegimeDescription

`ResponseRegimeDescription` resume o caráter espectral observado de uma condição
dinâmica por descritores operacionais baseados em limiares explícitos aplicados
a métricas já calculadas. A descrição é separada em dimensões independentes:
estrutura espectral, evolução temporal, identidade operacional de linhas e
confiabilidade da evidência. Cada descritor preserva critérios individuais,
operadores, thresholds, pesos, scores de suporte e oposição, métricas
indisponíveis, conflitos e limitações como clipping, baixa relação sinal/fundo,
alta variabilidade entre repetições, métricas ausentes e resolução limitada. O
resultado pode descrever uma resposta como dominada por linhas discretas,
espectro denso, banda larga, mista ou com ataque banda larga seguido de cauda
mais tonal, mas esses rótulos não são provas de linearidade, não linearidade,
caos, transição física de regime, identidade modal ou conversão para
`ModalMode`.

### ModalMode

`ModalMode` representa uma componente modal identificada ou acompanhada em uma
gravação. Seu papel científico é organizar uma observação relacionada à resposta
vibratória e acústica do idiofone, incluindo propriedades que possam ser
estimadas e suas incertezas quando aplicáveis.

### ModalCandidate

`ModalCandidate` representa uma trajetória espectral caracterizada que satisfez
critérios matemáticos e operacionais explicitamente configurados. Ele precede
qualquer interpretação como `ModalMode`; sua aceitação é reversível, preserva
os critérios avaliados e não comprova frequência natural, amortecimento ou
identidade modal física.

### PreImpactEvidence

`PreImpactEvidence` descreve se o nível rastreado de uma componente mudou em
janelas configuráveis antes e depois do impacto. Presença pré-impacto isolada
não implica irrelevância nem rejeição: uma componente preexistente pode ser
amplificada ou reexcitada. O resultado é operacional e não estabelece
causalidade física definitiva.

### Associação dentro da condição de excitação

Uma condição dinâmica (`pp`, `p`, `mf`, `f`, `ff` ou `unspecified`) é uma
categoria experimental, não uma medida física absoluta. Candidatos de
repetições com o mesmo rótulo podem formar agrupamentos operacionais por
compatibilidade frequencial e critérios opcionais auditáveis. A associação
preserva candidatos sem correspondência, impede mistura entre rótulos e não
promove agrupamentos a `ModalMode`.

### Associação entre condições dinâmicas adjacentes

Candidatos operacionais de duas condições dinâmicas nominalmente adjacentes
podem ser associados individualmente por critérios explícitos e auditáveis de
compatibilidade. A função de baixo nível aceita somente os pares `pp -> p`,
`p -> mf`, `mf -> f` e `f -> ff`; pares invertidos, rótulos iguais, saltos
nominais e comparações diretas `pp -> ff` não fazem parte desta política.

O resultado deve preservar uma partição completa dos candidatos: correspondidos
um-a-um, desaparecidos na condição superior ou emergentes na condição superior.
Custos, diferenças de frequência absoluta, relativa simétrica e logarítmica,
qualidade de ajuste, tau, ambiguidade, proximidade de limiar, margens e
evidência pré-impacto devem permanecer disponíveis como diagnósticos
operacionais. A ausência de correspondência confiável é um resultado válido e
não deve ser ocultada por matching forçado.

Indícios de possível divisão ou possível fusão podem ser registrados quando as
alternativas admissíveis são próximas, mas não resolvem automaticamente
associações um-para-muitos ou muitos-para-um. Esses diagnósticos não concluem
divisão ou fusão física e não promovem candidatos a `ModalMode`.

### Encadeamento de associações adjacentes

Resultados já calculados por associação adjacente podem ser encadeados ao longo
de uma subsequência contígua da ordem nominal `pp -> p -> mf -> f -> ff`. A
entrada deve preservar pares conectados e válidos, sem duplicatas, lacunas,
saltos, rótulos invertidos ou condições desconhecidas. A sequência aceita
subconjuntos contíguos como `p -> mf -> f`, mas não cria política principal de
associação direta entre condições não adjacentes.

Cada cadeia é uma sequência operacional de candidatos conectados por matches
adjacentes já aceitos. Candidatos sem match permanecem como cadeias unitárias;
cadeias parciais que começam como emergentes ou terminam como desaparecidas
mantêm esses contextos a partir dos contratos existentes. Trajetórias de
frequência, mudanças assinadas por passo, classificações operacionais de
deslocamento, custos locais, margens, ambiguidade e proximidade de limiar devem
permanecer auditáveis até o match adjacente original.

Contextos de possível split ou possível merge podem ser anexados a cadeias
quando um de seus nós aparece nos diagnósticos já existentes, mas a cadeia
continua linear. O encadeamento não cria árvores, não otimiza rotas globais, não
fecha lacunas por proximidade de frequência, não resolve divisão ou fusão e não
transforma uma cadeia operacional em modo físico comprovado, família modal,
identidade modal persistente ou `ModalMode`.

### ModalHypothesis

`ModalHypothesis` representa a avaliação operacional de uma única
`CrossConditionCandidateChain` contra critérios explícitos, configuráveis e
auditáveis. A cadeia operacional de candidatos não é, por si só, uma hipótese
aceita; uma hipótese aceita também não é um modo físico comprovado. O resultado
deve preservar evidências separadas de cobertura, continuidade frequencial,
qualidade de associação, tracking, consistência de tau, evidência pré-impacto e
contexto de possível split/merge, além de score normalizado, pesos, penalidades
e razões de suporte, ressalva, rejeição e ausência de evidência.

A decisão deve obedecer à precedência: entrada inválida, falha de gate
obrigatório, evidência insuficiente, rejeição por critérios configurados,
aceitação com ressalvas, aceitação e, por fim, inconclusão quando não houver
base suficiente para aceitar ou rejeitar. O score é uma métrica de auditoria e
não pode sobrepor critérios obrigatórios. Ausência de tau, evidência
pré-impacto, margens ou métricas de tracking deve permanecer explícita e ser
tratada conforme política configurável, nunca como zero implícito.

A camada não recalcula FFT, STFT, tracking, seleção de candidatos ou associação
entre condições; ela usa apenas candidatos já caracterizados, matches adjacentes
aceitos, cadeias já construídas e diagnósticos existentes. Ela não cria
associações não adjacentes, não fecha lacunas, não troca matches locais, não
otimiza rotas globais, não resolve split ou merge e não cria `ModalMode`.
Persistência entre condições, deslocamento de frequência e contexto de
split/merge permanecem evidências operacionais, não provas físicas de
identidade modal, não linearidade, divisão ou fusão.

### ModalParameterEstimate

`ModalParameterEstimate` representa uma síntese quantitativa operacional dos
valores já disponíveis em uma `ModalHypothesis`. A estimativa pode reunir
frequência representativa, trajetória frequencial entre condições, variação e
drift descritivos, tau representativo, taxa matemática de decaimento,
incertezas operacionais, cobertura, ressalvas, insuficiências, invalidades e
proveniência até candidatos, matches, cadeia e configuração.

A camada deve preservar explicitamente:

```text
hipótese modal != modo físico comprovado
frequência representativa != frequência modal exata
tempo de decaimento estimado != constante física invariável
variação entre condições != prova de não linearidade
incerteza operacional != intervalo de confiança físico completo
```

Os estados de uma estimativa são mutuamente exclusivos: `valid`,
`valid_with_reservations`, `partial`, `insufficient_evidence` e
`invalid_input`. A decisão deve seguir precedência explícita: entrada inválida,
hipótese não permitida, frequência insuficiente, tau insuficiente quando
exigido, violação crítica de dispersão, estimativa parcial, estimativa válida
com ressalvas e estimativa válida. O status não deve ser inferido apenas de
score.

A frequência representativa deve usar métodos de localização e peso declarados,
como média, mediana, média ponderada, mediana ponderada, pesos uniformes,
cobertura, qualidade de ajuste, custo inverso de associação ou combinação
documentada. Tau deve privilegiar o domínio logarítmico e só aceitar valores
estritamente positivos e finitos. Valores ausentes ou inválidos não podem ser
substituídos por zero.

Incertezas de frequência e tau são operacionais. Podem usar dispersão amostral,
erro padrão, MAD escalado, bootstrap percentil determinístico com seed explícita
ou combinações conservadoras quando incertezas individuais existirem. Esses
intervalos não são automaticamente intervalos de confiança físicos completos.

A taxa de decaimento deve declarar a convenção usada. Quando
`A(t) = A0 exp(-t / tau)`, a taxa de decaimento de amplitude é `1 / tau`; tempos
em dB devem derivar dessa mesma convenção e não devem ser confundidos com
decaimento de energia. A camada de parâmetros não calcula fator de qualidade
`Q`, largura de banda, ajuste físico de oscilador, acoplamento modal, troca de
energia, hardening, softening, causalidade, fechamento de lacunas, associação
não adjacente, resolução de split/merge ou promoção para `ModalMode`.

### ModalQFactorEstimate

`ModalQFactorEstimate` representa uma estimativa operacional de fator de
qualidade associada a uma `ModalParameterEstimate`. A camada pode combinar dois
métodos independentes quando ambos estiverem disponíveis:

- `Q_decay`, derivado de frequência representativa e tau representativo;
- `Q_bandwidth`, derivado de frequência central e largura de banda espectral.

Esses valores são condicionados às hipóteses do método e à configuração ativa.
A RFC declara explicitamente:

```text
hipótese modal != modo físico comprovado
Q estimado por decaimento != Q físico exato
Q estimado por largura de banda != Q físico exato
concordância entre métodos != validação física definitiva
discordância entre métodos != prova de erro ou não linearidade
```

A convenção de decaimento é a mesma da camada de parâmetros:
`A(t) = A0 exp(-t / tau)`. Para um resumo operacional de oscilador levemente
amortecido, a relação usada é `Q_decay = pi * f * tau`, em que `f` está em hertz
e `tau` é a constante de decaimento de amplitude em segundos. Essa relação deve
registrar suas hipóteses: decaimento aproximadamente exponencial, amortecimento
fraco, componente suficientemente isolada e frequência aproximadamente estável
no intervalo analisado. Tau de amplitude não deve ser confundido com tau de
energia.

A largura de banda deve declarar sua definição. A convenção padrão é largura
total a -3 dB em amplitude, isto é, nível de corte `1/sqrt(2)` da amplitude de
pico. Quando a largura vier de `SpectralPeak.width_hz` ou
`GlobalSpectralPeakMetric.width_hz`, a estimativa deve preservar a definição
original de meia proeminência, em amplitude do espectro de origem ou potência
linear canônica, respectivamente. A camada não deve misturar amplitude e
potência, não deve extrapolar cruzamentos, não deve aceitar largura zero e não
deve usar bins extremos como cruzamentos silenciosos.

O fator por largura usa somente a relação `Q_bandwidth = f_center / bandwidth`
com frequência e largura estritamente positivas e finitas. A resolução
espectral deve ser diagnosticada por `bandwidth_hz / frequency_resolution_hz`,
com classificação operacional como bem resolvido, marginal, limitado por
resolução ou não resolvido segundo limiares configuráveis. Picos vizinhos devem
ser diagnosticados por distância e fração de sobreposição, sem tentar separar
picos sobrepostos ou ajustar múltiplas Lorentzianas.

A comparação entre métodos deve usar diferença relativa simétrica,
`abs(Q1 - Q2) / ((Q1 + Q2) / 2)`, e diferença logarítmica com valores positivos.
Combinação de métodos só é permitida por política explícita, como média simples,
média geométrica, média ponderada por incerteza, preferência declarada ou
nenhuma combinação. Métodos inconsistentes não devem ser combinados por padrão.

Os estados são mutuamente exclusivos: `valid`, `valid_with_reservations`,
`partial`, `inconclusive`, `insufficient_evidence` e `invalid_input`. A decisão
deve seguir precedência explícita: entrada inválida, status de origem não
permitido, ausência dos dois métodos, método obrigatório ausente, método
disponível porém inválido, inconsistência forte entre métodos, método único
válido, métodos válidos com ressalvas, métodos válidos consistentes e resultado
válido. O status não deve ser determinado por score nem promover a estimativa a
`ModalMode`.

A camada não abre WAV, não recalcula FFT, STFT, tracking, candidatos ou matches,
não fecha lacunas, não cria associação não adjacente, não resolve split ou
merge, não infere hardening, softening, linearidade, não linearidade,
causalidade, troca de energia ou acoplamento modal.

### ModalEnergyExchangeEvidence

`ModalEnergyExchangeEvidence` representa somente evidência operacional
compatível com possível redistribuição aparente de energia entre dois
componentes ou hipóteses modais dentro de uma mesma gravação ou condição
dinâmica. A camada usa envelopes de amplitude, séries temporais de tracks,
frequências, tau, estimativas modais, diagnósticos e proveniência já
disponíveis. Ela não abre WAV, não recalcula FFT/STFT, não refaz tracking, não
recria candidatos ou associações, não fecha lacunas e não resolve split ou
merge.

A RFC declara explicitamente:

```text
anticorrelação entre envelopes != transferência física de energia comprovada
crescimento tardio != excitação interna comprovada
atraso temporal != causalidade
batimento aparente != acoplamento modal comprovado
conservação aproximada de soma de energias != sistema fechado
evidência operacional de troca != prova de troca física
```

A representação de amplitude deve ser configurada, podendo usar amplitude
linear, amplitude normalizada, amplitude em dB, potência relativa ou energia
operacional proporcional ao quadrado da amplitude. Quando usada, a grandeza
`E_proxy(t) proportional A(t)^2` deve ser chamada de proxy de energia,
energia operacional ou energia relativa aparente, nunca simplesmente energia
física. Ponderação por frequência deve ser explicitamente habilitada e
documentada.

O preparo de envelopes deve preservar tempos estritamente crescentes,
amplitudes finitas, máscara válida, normalização, suavização opcional,
interpolação e diagnósticos. Alinhamento temporal pode exigir eixos idênticos,
usar interseção, interpolação linear na faixa comum ou downsampling, mas não
pode extrapolar nem preencher lacunas longas silenciosamente. Valores ausentes
ou inválidos permanecem explícitos e não são substituídos por zero.

A evidência pode reunir tendências opostas, anticorrelação, correlação com lag,
crescimento tardio, recuperação de amplitude, alternância de dominância,
estabilidade aproximada do proxy de energia do par e contexto de possível
batimento. A convenção de lag é descritiva: `lag > 0` significa que mudanças no
componente A precedem mudanças no componente B no eixo alinhado; isso não
declara direção causal.

Significância operacional deve ser determinística, por exemplo por deslocamento
circular ou permutação em blocos com seed explícita. O score é uma soma
auditável de componentes configurados; gates obrigatórios prevalecem sobre o
score. Batimento aparente deve gerar ressalva ou contexto, não aumentar
automaticamente a evidência de redistribuição.

Os estados são mutuamente exclusivos: `supported`,
`supported_with_reservations`, `inconclusive`, `not_supported`,
`insufficient_evidence` e `invalid_input`. A decisão deve seguir precedência
explícita: entrada inválida, origem não permitida, envelope ausente,
sobreposição insuficiente, amostras insuficientes, gate obrigatório falho,
evidência insuficiente, evidência contraditória, ausência de suporte, suporte
com ressalvas e suporte. O status não deve declarar acoplamento, causalidade,
transferência física, linearidade, não linearidade, split, merge ou identidade
modal física.

### Validação sintética científica

`SyntheticValidationScenario` representa um cenário controlado cujo conteúdo é
conhecido por construção antes da análise. `SyntheticGroundTruth` preserva
sinal limpo, ruído, sinal observado, frequências verdadeiras sintéticas, tau,
Q pela convenção compatível `Q = pi * f * tau`, largura de banda conhecida
quando identificável, presenças, associações, cadeias e pares de evidência
operacional esperados. Essa verdade sintética não é derivada do resultado do
BellLab.

`SyntheticPipelineOutput` registra a execução das APIs públicas disponíveis:
análise temporal, FFT estacionária, detecção de picos, STFT, picos por quadro,
tracking, caracterização de candidatos e, quando as entradas existem, camadas
de associação, cadeias, hipóteses, parâmetros, Q e evidência operacional de
possível redistribuição de energia. Falhas de estágio devem permanecer
explícitas; uma execução parcial configurada não pode ocultar o estágio ausente.

As validações sintéticas comparam `verdade sintética conhecida` contra
`resultado recuperado pelo BellLab` por métricas de frequência, trajetória,
drift, tau, Q, largura de banda, tracking, candidatos, associações, cadeias,
hipóteses e evidência operacional de possível redistribuição de energia.
Campanhas e Monte Carlo usam seeds explícitas e ordem determinística. A camada
não deve calibrar thresholds usando o resultado da mesma realização, corrigir
tracking com a verdade, usar valores verdadeiros como entrada oculta dos
estimadores, resolver split/merge físico, declarar causalidade ou promover
qualquer sucesso sintético a `ModalMode`.

Os estados são mutuamente exclusivos: `passed`,
`passed_with_reservations`, `failed`, `inconclusive`,
`insufficient_evidence`, `invalid_scenario` e `pipeline_error`. Cenários
fundamentalmente não identificáveis podem ser classificados corretamente como
insuficientes ou inconclusivos. Recuperação correta em sinal sintético não
garante validade em dados reais; erro baixo em um cenário não implica robustez
geral; aprovação de threshold não é prova física; falha de recuperação não é
falha universal do método.

### Caracterização da condição de excitação

O rótulo musical de dinâmica descreve uma categoria ou intenção experimental,
não uma intensidade física absoluta. A condição de excitação pode preservar
metadados de microfone, interface, ganho, canal, distância, posição e excitador,
enquanto uma caracterização separada mede pico, RMS, energia discreta, duração
operacional, clipping, fundo e relação sinal/fundo na unidade efetivamente
disponível. Níveis dBFS exigem referência digital e não são níveis dB SPL.
Consistência ordinal entre `pp`, `p`, `mf`, `f` e `ff` é apenas um diagnóstico:
inversões não renomeiam gravações e não iniciam associação entre condições.

### Experiment

`Experiment` representa um contexto reprodutível de investigação. Ele reúne
gravações, configurações de análise, metadados metodológicos e resultados
produzidos em uma campanha ou pergunta científica definida.

`Experiment` também explicita uma relação comparativa entre gravações,
resultados, idiofones ou campanhas. Deve declarar referências, critérios de
compatibilidade e limitações, evitando comparações implícitas entre condições
incompatíveis.

---

## 5. Grandezas físicas

### 5.1 Pipeline reprodutível de experimento real

O BellLab define uma camada pública de orquestração para experimentos acústicos
reais por meio de `ExperimentDefinition`, `ExperimentRecordingDefinition`,
`ExperimentPipelineSettings` e `analyze_experiment(...)`. Essa camada recebe
somente caminhos e metadados explicitamente fornecidos, carrega WAVs pela API
canônica de I/O e coordena as camadas científicas já existentes:

```text
arquivos e metadados
→ carregamento
→ validação
→ análise temporal
→ espectro global
→ STFT
→ tracking
→ pré-impacto
→ excitação
→ candidatos
→ associação dentro da condição
→ associação entre condições adjacentes
→ cadeias
→ hipóteses modais
→ parâmetros
→ Q e largura de banda
→ evidência operacional de possível redistribuição de energia
→ resumo estruturado
```

O pipeline é um orquestrador. Ele não duplica a lógica científica dos módulos
subjacentes, não recalibra limiares usando o mesmo conjunto analisado e não
usa sucesso computacional como evidência física suficiente. Uma execução
concluída não prova validade física, identidade modal, linearidade,
não linearidade, causalidade, transferência física de energia, acoplamento,
hardening, softening ou resolução física de split/merge.

Cada estágio registra status terminal, dependências, entradas, saídas, razões,
diagnósticos e resultados intermediários quando disponíveis. Resultados
parciais, falhas, bloqueios, insuficiências e estágios omitidos devem permanecer
visíveis. `None` deve continuar representando ausência real de valor; o
pipeline não deve fabricar zeros para metadados, estimativas ou medições
ausentes.

Condições dinâmicas seguem a ordem canônica:

```text
pp → p → mf → f → ff
```

Associações de candidatos entre condições são permitidas somente entre pares
nominalmente adjacentes presentes. Se uma condição intermediária estiver
ausente, a lacuna deve ser diagnosticada explicitamente; o pipeline não deve
associar automaticamente condições não adjacentes, não deve fechar lacunas e
não deve construir cadeias atravessando a lacuna.

Múltiplas repetições de uma mesma condição são analisadas separadamente. A
seleção de uma gravação de referência, quando configurada, deve ser auditável e
baseada em política explícita; o pipeline não deve calcular média de formas de
onda nem misturar sinais brutos por padrão. Seleção de canal, offsets e
polarity devem ser explícitos, preservando caminho original, duração original,
duração analisada, fingerprints de conteúdo e metadados disponíveis sem
inventar informações desconhecidas.

### 5.5 Exportação reproduzível de resultados

O BellLab define uma camada pública de exportação reproduzível para resultados
já calculados, em especial `ExperimentAnalysisResult`. Essa camada transforma
contratos científicos existentes em representações estruturadas, ordenadas e
auditáveis, como JSON, CSV, fragmentos LaTeX, Markdown e manifesto de
proveniência. A exportação preserva IDs, status, incertezas, ausências,
ressalvas, invalidades, diagnósticos, configurações, fingerprints de arquivos e
versão do BellLab.

Exportação bem-sucedida não implica validade científica do resultado. Uma tabela
formatada não é evidência física suficiente, um valor ausente não é zero, uma
hipótese modal não é um modo físico comprovado e evidência operacional de
possível redistribuição de energia não é transferência física comprovada. A
camada de exportação não deve recalcular espectros, tracking, candidatos,
hipóteses, parâmetros, Q ou energia operacional, nem alterar os resultados de
origem para melhorar a apresentação.

Arquivos exportados devem ter nomes determinísticos, política explícita de
sobrescrita, escrita atômica quando configurada, checksums de conteúdo e
manifesto escrito por último. JSON deve ser padronizado e não deve conter tokens
não finitos como `NaN` ou `Infinity`; valores não finitos exigem política
explícita. CSVs devem ser normalizados em tabelas com cabeçalhos estáveis,
chaves primárias e chaves estrangeiras preservadas. Tabelas LaTeX e Markdown
são camadas de apresentação e não devem arredondar ou modificar os valores
armazenados no modelo normalizado.

O BellLab poderá futuramente calcular, armazenar, comparar ou relatar as
seguintes grandezas, entre outras cientificamente justificadas:

- tempo e duração;
- frequência;
- amplitude;
- fase;
- energia;
- potência;
- nível de pressão sonora, quando houver calibração apropriada;
- nível relativo em dBFS, quando apropriado;
- ruído e piso de ruído;
- relação sinal-ruído;
- faixa dinâmica;
- instante de impacto;
- envelope de amplitude;
- constante de decaimento;
- tempo característico para 1/e;
- tempo característico para -20 dB;
- tempo característico para -40 dB;
- tempo característico para -60 dB;
- largura de banda;
- frequência central;
- fator de qualidade Q;
- razão de amortecimento;
- energia modal;
- estabilidade, deriva e rastreamento de frequência modal;
- relações entre modos;
- métricas de similaridade e diferença entre gravações;
- incertezas, intervalos de confiança e indicadores de qualidade, quando
  pertinentes.

A presença de uma grandeza em um resultado deve indicar sua unidade, seu
contexto de análise e sua disponibilidade. Esta RFC não define, nesta etapa, os
métodos de cálculo ou critérios de validade de cada grandeza.

---

## 6. Convenções

O BellLab adota as seguintes convenções de projeto e comunicação científica:

- As unidades devem seguir o Sistema Internacional de Unidades (SI) sempre que
  aplicável.
- Tempo deve ser expresso em segundos (`s`).
- Frequência deve ser expressa em hertz (`Hz`).
- Amplitude deve ser expressa em dBFS quando apropriado à natureza digital e não
  calibrada da gravação; unidades físicas devem ser declaradas somente quando a
  calibração correspondente estiver disponível.
- Toda unidade, escala, referência e transformação de amplitude deve ser
  declarada junto ao resultado.
- A nomenclatura de objetos, campos, grandezas e relatórios deve ser consistente
  em todo o projeto.
- O espaçamento entre bins de uma FFT deve ser distinguido da resolução
  espectral efetiva; zero padding densifica a grade, mas não cria informação
  nova nem define sozinho a separabilidade de componentes próximas.
- Métricas de distribuição de energia devem usar potência ou densidade de
  potência linear declarada, nunca valores em decibéis diretamente. Conversões
  de dB exigem referência recuperável e devem permanecer auditáveis.
- Flatness, entropia, densidade de picos e energia residual são descritores
  globais, não classificadores automáticos de ruído ou regime físico.
- Tendências temporais, regiões early/middle/late e change points de métricas
  espectrais são descritores operacionais de evolução, não provas de transição
  de regime físico, não linearidade, caos ou identificação modal.
- Um pico espectral é uma observação matemática e não deve ser interpretado
  automaticamente como modo físico do instrumento.
- Resultados devem ser reproduzíveis a partir das entradas, versões, parâmetros
  e métodos registrados.
- Ausência de dados, estimativas indisponíveis e limitações de qualidade devem
  ser representadas explicitamente, sem serem convertidas em valores implícitos.
- O núcleo do BellLab deve permanecer independente do tipo de instrumento. As
  diferenças entre instrumentos devem ser implementadas apenas através de
  especializações de domínio, nunca através de alterações nos algoritmos
  fundamentais.

---

## 7. Requisitos científicos

Todo algoritmo incorporado ao BellLab deverá:

- informar seus parâmetros de entrada e configuração;
- registrar o método utilizado e sua versão;
- permitir a reprodução completa dos resultados a partir das entradas
  disponíveis;
- produzir metadados suficientes para auditoria científica;
- declarar unidades, escalas e referências dos valores produzidos;
- registrar condições relevantes de validade, qualidade e limitações;
- preservar a ligação entre resultado, sinal de origem e contexto experimental;
- evitar resultados silenciosamente ambíguos ou sem proveniência.

Os relatórios e exportações devem carregar informação suficiente para que um
leitor possa identificar o conjunto de dados, as configurações e a cadeia de
análises que conduziram a cada conclusão apresentada.

---

## 8. Extensibilidade

A arquitetura científica do BellLab deve permitir evolução sem romper a
reprodutibilidade ou a interpretação dos resultados existentes. Entre as
extensões previstas estão:

- comparação entre idiofones;
- comparação entre campanhas de aquisição;
- análise de conjuntos de idiofones;
- análise de carrilhões e suas relações de conjunto;
- análise integrada de vibração estrutural;
- correlação entre dados acústicos, históricos, geométricos e de conservação;
- exportação estruturada para artigos científicos, anexos de dados e materiais
  suplementares;
- integração com repositórios de dados e fluxos de revisão científica.

Qualquer extensão deve respeitar as convenções desta RFC e introduzir contratos
de dados explícitos antes de acrescentar novas análises. Quando uma extensão
alterar o significado científico de resultados já definidos, ela deverá ser
documentada em uma RFC complementar.
