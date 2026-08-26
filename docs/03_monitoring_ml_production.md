# Monitoring de modèles en production

## Le problème : un modèle se périme

Une API web qui répond `200` fonctionne. Un modèle ML qui répond `200` peut se tromper systématiquement depuis trois semaines sans qu'aucune alerte ne se déclenche. C'est la difficulté propre au monitoring ML : la panne est silencieuse.

La cause est presque toujours la même. Le modèle a appris une relation entre des données d'entrée et une cible à un instant donné. Le monde, lui, continue de bouger.

## Data drift et concept drift

Le **data drift** est un changement de distribution des variables d'entrée. Votre modèle de scoring crédit a été entraîné sur une clientèle dont l'âge moyen était de 42 ans ; une campagne marketing amène des utilisateurs de 25 ans. Les données ont changé, la relation apprise reste peut-être valide, mais le modèle extrapole hors de son domaine d'entraînement.

Le **concept drift** est plus insidieux : c'est la relation entre entrées et cible qui change. Un modèle de détection de fraude entraîné avant l'apparition d'une nouvelle technique d'attaque voit les mêmes variables d'entrée, mais leur signification a changé. Les distributions peuvent être parfaitement stables pendant que la performance s'effondre.

Détecter le data drift ne demande que les prédictions. Détecter le concept drift demande la **vérité terrain**, qui arrive souvent avec des semaines de retard — un remboursement de crédit se constate sur des mois. C'est pourquoi on surveille les deux.

## Quatre familles de métriques

**Métriques techniques** — latence p95, taux d'erreur, débit. Elles ne disent rien de la qualité des prédictions, mais un modèle indisponible est un modèle inutile.

**Métriques de données** — taux de valeurs manquantes par colonne, part de catégories inconnues, distance entre la distribution d'entraînement et celle de production. Le test de Kolmogorov-Smirnov pour les variables continues et l'indice PSI (Population Stability Index) sont les outils standards. Une convention répandue : PSI < 0,1 stable, 0,1–0,25 à surveiller, > 0,25 drift avéré.

**Métriques de prédiction** — distribution des sorties du modèle. Si un classifieur binaire prédisait 8 % de positifs et en prédit soudain 30 %, quelque chose a changé en amont. C'est un signal précoce et gratuit, disponible sans vérité terrain.

**Métriques métier** — accuracy, F1, AUC calculés une fois les labels réels disponibles, et surtout l'impact business : taux de conversion, fraudes réellement bloquées.

## Outils

**Evidently** est une bibliothèque Python open source qui compare un jeu de référence et un jeu courant, puis produit un rapport de drift en HTML ou en JSON exploitable dans un pipeline.

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_train, current_data=df_prod)
report.save_html("drift_report.html")
```

**Prometheus + Grafana** couvrent le temps réel : l'application expose ses métriques, Prometheus les collecte, Grafana les affiche. C'est le socle standard côté infrastructure.

## Alerting : moins, mais mieux

Une alerte doit être actionnable. « PSI > 0,25 sur la variable `revenu` pendant 24 h » indique quoi regarder ; « anomalie détectée » ne sert à rien.

Deux règles de survie. D'abord, seuiller sur une fenêtre temporelle et non sur un point isolé, sinon un pic de trafic déclenche une astreinte inutile. Ensuite, hiérarchiser : une alerte critique réveille quelqu'un la nuit, une alerte de drift alimente un rapport hebdomadaire. Une équipe qui reçoit trente alertes par jour finit par toutes les ignorer, y compris la bonne.
