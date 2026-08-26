# CI/CD pour les pipelines ML

## Pourquoi automatiser ?

Un projet ML accumule vite les étapes manuelles : lancer les tests, réentraîner sur les nouvelles données, vérifier que la métrique ne s'est pas dégradée, construire l'image Docker, déployer. Fait à la main, ce rituel est oublié un vendredi soir, et un modèle défaillant part en production.

La **CI** (intégration continue) exécute automatiquement les vérifications à chaque push. La **CD** (déploiement continu) pousse en production ce qui a passé les vérifications. L'objectif n'est pas la vitesse : c'est de rendre les erreurs visibles tôt et de rendre le déploiement ennuyeux.

## Une CI GitHub Actions minimale

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

Ce fichier va dans `.github/workflows/ci.yml`. À chaque push, GitHub provisionne une machine, installe les dépendances et lance les tests. Un échec bloque la pull request.

## Que teste-t-on sur un projet ML ?

Les tests unitaires classiques s'appliquent à votre code de feature engineering : une fonction de normalisation doit produire une moyenne nulle, un parseur de dates doit gérer les valeurs manquantes. C'est du logiciel ordinaire, testez-le comme tel.

Viennent ensuite les tests spécifiques au ML, souvent négligés :

- **Test de schéma** : le dataset d'entrée contient-il les colonnes attendues, dans les bons types ?
- **Test de non-régression de performance** : le modèle fraîchement entraîné atteint-il au moins le F1 du modèle en production ? Si la métrique chute de 5 points, on veut un échec rouge, pas un déploiement.
- **Test de forme** : le modèle sérialisé se recharge-t-il et renvoie-t-il une prédiction de la bonne dimension pour un input de référence ?

```python
def test_model_beats_baseline():
    model = joblib.load("artifacts/model.pkl")
    score = f1_score(y_test, model.predict(X_test))
    assert score >= 0.82
```

Ce seuil codé en dur est volontairement rustique, mais il attrape déjà l'essentiel des régressions.

## Du test au déploiement

Une fois les tests verts, la CD construit l'image Docker, la tague avec le SHA du commit et la pousse sur un registry :

```yaml
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t registry.exemple.com/api:${{ github.sha }} .
      - run: docker push registry.exemple.com/api:${{ github.sha }}
```

Le `needs: test` garantit qu'on ne déploie jamais du code non testé, et la condition sur `main` évite de déployer chaque branche de feature.

Taguer avec le SHA plutôt qu'avec `latest` est important : on sait exactement quel commit tourne en production, et un rollback consiste simplement à redéployer le tag précédent.

## Le piège du réentraînement automatique

Réentraîner à chaque commit est tentant, mais coûteux et rarement utile. La pratique courante est de séparer deux pipelines : la CI code tourne à chaque push (minutes), le pipeline de réentraînement tourne sur un déclencheur explicite — planification hebdomadaire, ou alerte de drift détectée en production.
