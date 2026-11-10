
# Classification Pipeline — Gênero musical de faixas do Spotify

### Pergunta

É possível classificar o gênero musical de uma faixa usando apenas seus atributos de áudio e o ano de lançamento, com os modelos da [ML-Library](https://github.com/Machine-Learning-Name-in-progress/ML-Library)?

- **Alvo:** `playlist_genre`, com 6 classes: `edm`, `latin`, `pop`, `r&b`, `rap` e `rock`.
- **Entradas:** 11 atributos numéricos do Spotify (`danceability`, `energy`, `loudness`, `tempo`, `track_popularity` etc.), `key`, `mode` e o ano de lançamento extraído da data do álbum.
- **Métrica principal:** F1-macro, que dá o mesmo peso aos 6 gêneros. A acurácia é reportada como complemento.

## Dados

`data/spotify_songs.csv`: conjunto [30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) (Kaggle). São 32.833 linhas e 23 colunas; cada linha é uma faixa dentro de uma playlist, então a mesma faixa pode aparecer mais de uma vez (28.356 `track_id` distintos).

| Coluna                                                                                                           | Conteúdo                                                   |
| ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `track_id`, `track_album_id`, `playlist_id`                                                                | Identificadores do Spotify                                  |
| `track_name`, `track_artist`, `track_album_name`, `playlist_name`                                        | Nomes (texto)                                               |
| `track_album_release_date`                                                                                     | Data de lançamento do álbum (AAAA-MM-DD, AAAA-MM ou AAAA) |
| `playlist_genre`                                                                                               | Gênero da playlist (alvo)                                  |
| `playlist_subgenre`                                                                                            | Subgênero da playlist                                      |
| `track_popularity`                                                                                             | Popularidade da faixa (0–100)                              |
| `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` | Atributos de áudio calculados pelo Spotify (0–1)          |
| `key`                                                                                                          | Tom (0 = Dó, …, 11 = Si)                                  |
| `mode`                                                                                                         | Modo (1 = maior, 0 = menor)                                 |
| `loudness`                                                                                                     | Volume médio (dB)                                          |
| `tempo`                                                                                                        | Andamento (BPM)                                             |
| `duration_ms`                                                                                                  | Duração (ms)                                              |

O gênero é o da playlist, não da faixa: o rótulo tem ruído, e `playlist_subgenre` e `playlist_name` praticamente determinam o alvo. Por isso essas colunas ficam fora do modelo.

## Como executar

Requisitos: Python 3.11 ou superior e [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Machine-Learning-Name-in-progress/Classification-Pipeline.git
cd Classification-Pipeline
uv sync
```

## Etapas do notebook

| Seção                  | Conteúdo                                                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 0. Configuração        | Imports, semente e constantes globais                                                                                                          |
| 1. Problema e dados      | Pergunta, métrica principal e dicionário de variáveis                                                                                       |
| 2. EDA                   | Valores ausentes, faixas duplicadas e com mais de um gênero, distribuição do alvo e das variáveis, relação com o gênero e vazamento     |
| 3. Divisão treino/teste | Remoção de faixas duplicadas e com mais de um gênero; divisão estratificada 80/20                                                          |
| 4. Pré-processamento    | Remoção de identificadores, textos e colunas com vazamento; ano de lançamento; one-hot de `key`; padronização com parâmetros do treino |
| 5. Modelos candidatos    | Classe majoritária (baseline), LDA, QDA, regressão logística (um-contra-todos) e KNN                                                        |
| 6. Validação           | Validação cruzada com 5 folds e busca de hiperparâmetros (`reg_param` e `k`)                                                            |
| 7. Comparação          | F1-macro médio por modelo e escolha do modelo final                                                                                           |
| 8. Avaliação final     | Métricas no teste (gerais e por gênero) e matriz de confusão                                                                                |
| 9. Conclusão            | Interpretação e limitações                                                                                                                 |

## Reprodutibilidade

- **Biblioteca:** a ML-Library é instalada a partir do GitHub (`[tool.uv.sources]` no `pyproject.toml`), com o commit travado no `uv.lock`.
- **Semente:** `SEED = 67`, aplicada com `set_seed` da biblioteca e usada em todos os sorteios do notebook.
- **Divisão estratificada:** 20% de cada gênero vão para o teste (5.333 faixas) e o restante para o treino (21.335), que é embaralhado antes da validação cruzada.

## Licença

[MIT](LICENSE)
