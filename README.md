# Lunar Lander with LTLf Reward Shaping

Repository di lavoro per esperimenti di reinforcement learning su LunarLander
con reward shaping guidato da automi.

## Framework attivi

La repository mantiene due soli framework:

| Directory | Uso |
| --- | --- |
| `multilevel_framework` | Framework multi-livello con un singolo learner |
| `manual_experiment` | Framework manuale per task ciclici |

Ogni framework contiene i sorgenti in `src/`, i launcher Docker nella propria
root e scrive i nuovi artefatti in `results/<experiment-name>/`.

## Avvio

Dalla root della repository:

```bash
./multilevel_framework/run_experiment.sh \
  --experiment-name multilevel-base \
  --episodes 1000 --num-seeds 5 --seed 42

./manual_experiment/run_experiment.sh \
  --experiment-name manual-cycle \
  --episodes 1000 --num-seeds 5 --seed 42
```

I launcher costruiscono l'immagine Docker del framework, verificano la
disponibilita di CUDA e montano la directory del framework in `/workspace`.
Gli script `run_evaluation.sh` usano lo stesso ambiente per valutare policy
gia addestrate.

## Configurazioni e notebook

Le configurazioni riutilizzabili sono raccolte in `templates/`:

- `templates/abstractions/` per le gerarchie multi-livello;
- `templates/trajectories/` per i task sequenziali;
- `templates/cyclic/` per i task ciclici.

I task definiscono ogni proposizione come una regione circolare continua nelle
coordinate normalizzate di LunarLander (`x` in `[-1, 1]`, `y` in `[0, 1.5]`):

```json
"regions": {
  "goal": {"center": [0.42, 1.05], "radius": 0.12}
}
```

Training e valutazione verificano l'appartenenza sullo stato continuo. Per il
planning astratto, ogni regione etichetta tutte le celle che interseca; ogni
livello della gerarchia viene rasterizzato direttamente dalla regione continua.

I notebook attivi sono:

- `notebook/notebook_multilevel_framework.ipynb`;
- `notebook/notebook_manual_experiment.ipynb`.

Per riallinearne le celle ai sorgenti correnti:

```bash
python3 notebook/sync_training_notebooks.py
```

Lo script `experiment_comparison.py` rimane nella root per generare grafici e
confrontare i risultati dei nuovi esperimenti.

## Materiale accantonato

Le implementazioni multi-epsilon, DSAC/SAC, dual-learner, i relativi notebook
e tutti gli esperimenti preesistenti sono conservati senza eliminazioni in
`backup/legacy_2026-08-19/`. Il manifesto in quella directory descrive il
contenuto e indica come ripristinarlo.
