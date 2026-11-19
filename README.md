# X-Bow Control

Treino de X-Bow 3.0 e IceBow com partidas vencedoras do IL_Replay e self-play no RoyaleSim. Os dois decks têm checkpoints próprios.

## Painel de treino e Modo Batalha

No PowerShell, abra a pasta do projeto e execute:

```powershell
.\.venv\Scripts\python.exe main.py
```

O comando sobe o painel de treino e batalha, imprime a URL no terminal e abre o navegador. A porta padrão é 8766. Se ela estiver ocupada, o painel abre em outra porta livre. Use `--no-open` para só imprimir a URL ou `Ctrl+C` no terminal para fechar a instância.

### 🎮 Jogar contra a IA

Acesse a seção **Jogar contra a IA** no painel aberto pelo comando.
- **Escolha seu deck**: Corredor 2.6, Gigante Real, Splashyard, X-Bow 3.0 ou IceBow.
- **Escolha o oponente (IA)**: X-Bow 3.0 Final, IceBow Final ou Rival Live. Os modelos finais carregam o PPO mais recente salvo de cada deck. O Rival Live carrega o checkpoint adversário mais recente do treino conjunto e pode jogar com Corredor 2.6, Gigante Real ou Splashyard.
- **Ajuste a velocidade** de 0,5× a 1,5×. A velocidade também pode ser ajustada durante a batalha.
- **Controles de batalha**:
  - Arraste uma carta para uma casa destacada ou selecione a carta e clique ou toque na arena.
  - No teclado, use `1`–`4` para selecionar, as setas para mirar e `Enter` para posicionar.
  - Abaixo da arena ficam a mão, a próxima carta e a barra de elixir.
  - Pressione `Espaço` para pausar ou retomar e `Esc` para cancelar a seleção.
  - Efeitos sonoros sintetizados pelo navegador e feedback tático em tempo real.

Abra **Cartas e decks** no menu para consultar, em uma página própria, a composição e o custo médio dos decks. Use o seletor para ver as oito cartas de um deck ou o catálogo completo do simulador. **Usar na batalha** volta ao painel com o deck selecionado.

No painel local, escolha X-Bow 3.0 ou IceBow. Em **Treinamento assistido**, escolha **Novo treino** ou selecione um experimento com checkpoints para **Retomar self-play**. Ajuste a duração e clique no botão de início. A tela acompanha a reconstrução dos replays, as épocas de imitação, as partidas de self-play e a avaliação, além de mostrar os registros e as métricas conforme são gravados.

O máximo do slider **Replays** acompanha a quantidade de candidatos do deck selecionado nas partições Parquet locais. O portal recalcula esse máximo quando as partições mudam. O simulador pode descartar replays durante a reconstrução.

**Parar treino** solicita a parada no próximo ponto seguro. Os checkpoints já gravados ficam disponíveis. Fechar a página não encerra o treino, que pode ser acompanhado ao reabrir o painel.

O simulador agrupa as previsões das partidas em andamento no mesmo processo. O treino PPO em CPU usa até quatro threads do PyTorch por padrão e roda até oito partidas em paralelo quando há pelo menos quatro threads. Nos demais casos, o padrão é quatro partidas em paralelo. O controle **Partidas por atualização** define o total de partidas antes de cada atualização PPO. Na linha de comando, use `--threads` para ajustar o PyTorch e `--parallel-games` de 1 a 8 para definir quantas partidas avançam juntas.

Em **Arena do treino**, veja o estado do tabuleiro, as mãos, o elixir e as jogadas registrados pelo simulador. **Imitação** compara a jogada humana com a decisão do modelo no mesmo quadro e mostra como a previsão muda ao longo dos lotes. **Self-play** e **Avaliação** mostram as partidas jogadas pelo modelo. Pause, avance pela gravação ou selecione um dos três registros mais recentes de cada etapa. Os exemplos com a indicação **Demonstração com dados locais** usam replays do dataset e checkpoints já salvos.

## Preparação no Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Coloque as [partições de replays](https://huggingface.co/datasets/VanguardX101/IL_Replay/tree/main/replays) em `replays/`. O parâmetro `--parquet-path` também aceita uma partição individual ou a pasta que contém `replays/`.

O importador `xbow_import_cochon.py` lê o snapshot publicado do [Cochon123](https://huggingface.co/datasets/Cochon123/clash-royale-replays) e salva em `replays/part-cochon-2026-11-01.parquet` as partidas vencedoras dos decks exatos que foram reconstruídas pelo simulador. Ele verifica a sequência de ações, remove partidas duplicadas e usa nível 11 quando o registro não informa os níveis.

Em **Vídeos do Royale TV**, o portal usa o índice local do [AlphaClash](https://huggingface.co/datasets/josefbednar/alphaclash-replays). O player carrega a gravação escolhida diretamente da fonte. As gravações do catálogo são de outra coleção e não compartilham identificadores com o IL_Replay.

## Imitação

```powershell
.\.venv\Scripts\python.exe xbow_dual_learning_pipeline.py --deck xbow_30 --parquet-path replays --scan
.\.venv\Scripts\python.exe xbow_dual_learning_pipeline.py --deck xbow_30 --parquet-path replays --matches 48 --epochs 3 --output runs/novo/xbow_30
.\.venv\Scripts\python.exe xbow_dual_learning_pipeline.py --deck icebow --parquet-path replays --matches 36 --epochs 3 --output runs/novo/icebow
```

O leitor identifica o vencedor a partir do resultado da partida e compara as oito cartas-base com o deck escolhido. O RoyaleSim reconstitui o tabuleiro a partir dos eventos gravados. A ordem inicial da mão é inferida pela sequência de cartas jogadas. O programa informa replays recusados por divergência de simulação, carta indisponível ou habilidade ambígua.

O treino aplica a reamostragem da equação 8 do [KataCR](https://arxiv.org/html/2504.04783v1) aos quadros de decisão. A divisão por partida usa o identificador do replay. As frequências de posicionamento por carta são estimadas só nas partidas de treino. O menor NLL de validação seleciona `bc.pt`, que também guarda a lista de partidas de cada conjunto. As métricas de carta e casa são condicionais aos quadros em que a pessoa jogou uma carta.

A perda de jogar ou esperar corrige a probabilidade de inclusão dos quadros reamostrados. A seleção da carta e da casa continua aprendendo dos quadros com jogadas reamostrados. Para ampliar o treino com a mesma validação, passe `--matches 120 --disk-frames --holdout-checkpoint runs/tactical/xbow_30/bc.pt` e escolha uma pasta nova de saída. `--disk-frames` mantém os quadros temporariamente no disco durante o treino.

## Self-play e avaliação

```powershell
.\.venv\Scripts\python.exe xbow_dual_learning_pipeline.py --deck xbow_30 --bc-checkpoint runs/novo/xbow_30/bc.pt --output runs/novo/xbow_30 --ppo-updates 4 --rollout-games 6 --kl-coef 0.25 --evaluate 6
.\.venv\Scripts\python.exe xbow_dual_learning_pipeline.py --deck xbow_30 --bc-checkpoint runs/novo/xbow_30/bc.pt --output runs/novo/xbow_30 --resume-ppo --ppo-updates 4 --rollout-games 6 --kl-coef 0.25
.\.venv\Scripts\python.exe xbow_dual_learning_pipeline.py --deck xbow_30 --bc-checkpoint runs/novo/xbow_30/bc.pt --output runs/novo/xbow_30 --evaluate-checkpoint runs/novo/xbow_30/ppo.pt --evaluate 12
.\.venv\Scripts\python.exe xbow_dual_learning_pipeline.py --deck xbow_30 --parquet-path replays --bc-checkpoint runs/novo/xbow_30/bc.pt --output runs/novo/xbow_30 --evaluate-checkpoint runs/novo/xbow_30/ppo.pt --replay-eval
```

Para IceBow, use `--deck icebow` e seus caminhos correspondentes. A liga usa checkpoints anteriores, um bot de X-Bow programado e rivais de Corredor e Gigante Real. O PPO penaliza a divergência KL exata do checkpoint congelado de imitação para a política em treino sobre as ações legais. O RoyaleSim fornece o resultado e o dano às torres, enquanto os termos adicionais recompensam aquisições de alvo pela X-Bow e trocas de elixir.

Para treinar duas IAs na mesma partida, em **Retomar PPO** de X-Bow 3.0 marque **Treinar X-Bow e a IA adversária na mesma partida**. A X-Bow continua do checkpoint selecionado e usa seu deck fixo. Uma única rede adversária alterna a cada partida entre os decks completos de Corredor 2.6, Gigante Real e Splashyard. As duas redes jogam, recebem as recompensas do próprio lado e atualizam seus pesos antes do próximo confronto. O checkpoint PPO salva as duas redes e seus otimizadores juntos após cada partida e a retomada continua da posição salva na rotação.

Na linha de comando, acrescente `--coevolve` ao comando de retomada de X-Bow 3.0. `--ppo-updates` indica o número de partidas. A avaliação opcional joga `--evaluate` partidas por deck adversário com pesos congelados e alterna os lados.

O treino conjunto usa a recompensa `potential` por padrão. Ela combina o resultado da partida com sinais de coroas, vida das torres e elixir na barra e nas tropas vivas. O foco da X-Bow aparece nas métricas da partida. `--reward legacy` seleciona a fórmula anterior para retomar um treino de controle.

Para comparar as duas fórmulas, execute o comando abaixo com a pasta de um checkpoint conjunto. Cada modelo treinado enfrenta o oponente inicial congelado com os três decks, usando as mesmas sementes e os dois lados. Use um número par para `--evaluate`, que vale por deck e por modelo. O relatório `reward_comparison.json` registra vitórias, dano às torres e elixir desperdiçado estimado pelo simulador, além das contribuições de recompensa. As duas variantes são salvas na nova pasta de comparação.

```powershell
.\.venv\Scripts\python.exe xbow_dual_learning_pipeline.py --coevolve --resume-ppo --output runs/calibrated/xbow_30 --bc-checkpoint runs/calibrated/xbow_30/bc.pt --ppo-updates 60 --evaluate 12 --compare-rewards runs/reward-ab
```

`selfplay.json` guarda as métricas do treino e da avaliação solicitada. Cada atualização registra o tempo de simulação (`simulation_seconds`) e de preparação e otimização do PPO (`optimization_seconds`). `evaluation.json` guarda avaliações executadas sem treino. Para comparar BC e PPO, o programa usa os mesmos rivais, sementes e lados. `replay_validation.json` compara a imitação antes e depois do PPO nas partidas reservadas e mede a escolha da carta e o posicionamento de X-Bow, Tesla e cartas de um elixir.

## Inferência local

```powershell
.\.venv\Scripts\python.exe xbow_policy_service.py --deck xbow_30 --checkpoint runs/tactical/xbow_30/ppo.pt
```

O processo lê um objeto JSON por linha com `spatial`, `vector` e `action_mask` no formato de `xbow_environment.battle_env()`. Ele retorna a ação e, quando aplicável, a posição na mão, a casa da arena ou o botão de habilidade.

### Entrada de um leitor de tela

`--public-frames` recebe quadros cronológicos com mão e elixir próprios, torres, unidades visíveis, feitiços e jogadas confirmadas. O primeiro quadro da partida usa tick 0. Cada segundo corresponde a 20 ticks. Cada quadro carrega eventos novos entre seu antecessor e o tick atual, com limite superior exclusivo. As coordenadas têm origem no canto inferior esquerdo da arena do observador, em tiles. A vida das torres vem na ordem rei, esquerda e direita de cada jogador. O formato está definido em `Frame` no módulo `xbow_public_feed.py`.

O adaptador abaixo lê o endpoint `/api/frames` de uma sessão de captura do cr-bot. Inicie a sessão antes da abertura. O relógio usa o tempo de captura e a vida máxima das torres usa os valores padrão do treino, que podem ser corrigidos por `--own-max` e `--opponent-max`. As sugestões usam os encaixes calculados a partir do tabuleiro público e excluem jogadas que dependem de realocação. O processo retorna sugestões em JSON. [API de quadros do cr-bot](https://github.com/Keschler/cr-bot/blob/main/src/frontend/api/frames.py)

```powershell
.\.venv\Scripts\python.exe -u xbow_capture_bridge.py --deck xbow_30 --capture-url http://127.0.0.1:8000 |
    .\.venv\Scripts\python.exe -u xbow_policy_service.py --deck xbow_30 --checkpoint runs/calibrated/xbow_30/bc.pt --public-frames
```

O adaptador também lê exportações cronológicas de quadros em JSONL pela entrada padrão. O rastreador descarta os campos de elixir estimado e cartas vistas fornecidos pelo leitor e refaz a estimativa com os eventos confirmados. A vida de unidades sem barra legível usa o valor base da carta no simulador.

O simulador fornece as jogadas públicas aceitas. O rastreador estima o elixir e a fila do rival com essas jogadas e o relógio de partida. Para verificar o contrato de observação e a recompensa de aquisição de alvo, execute `python -m unittest -v test_xbow_pipeline` no ambiente do projeto.
