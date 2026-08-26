# Évaluer un système LLM / RAG

## Pourquoi c'est plus difficile qu'une classification

Sur un problème de classification, la réponse est juste ou fausse et l'accuracy se calcule en une ligne. Sur un système RAG, la sortie est un texte libre : plusieurs formulations différentes peuvent être également correctes, et une réponse fluide et bien écrite peut être entièrement inventée.

D'où la première règle : **une démo qui « a l'air de marcher » n'est pas une évaluation**. Sans jeu de test, vous ne saurez pas si votre changement de stratégie de chunking a amélioré ou dégradé le système.

## Décomposer le pipeline

Un RAG a deux étages, et il faut les évaluer séparément — sinon on ne sait pas lequel corriger.

L'étage **retrieval** cherche les documents pertinents. L'étage **génération** rédige une réponse à partir de ces documents. Une mauvaise réponse peut venir d'un retrieval qui n'a pas trouvé le bon passage, ou d'un LLM qui avait le bon passage et l'a mal exploité. Ce sont deux corrections complètement différentes.

## Les métriques de retrieval

**Recall@k** : parmi les `k` documents remontés, le document contenant la réponse est-il présent ? C'est la métrique la plus importante, car si le passage n'est pas remonté, aucun prompt ne sauvera la génération.

**Precision@k** : quelle proportion des `k` documents remontés est réellement pertinente ? Un contexte pollué par du bruit dégrade la réponse et coûte des tokens.

**MRR** (Mean Reciprocal Rank) : à quelle position apparaît le premier document pertinent ? Utile quand l'ordre compte, les LLM accordant plus de poids au début du contexte.

## Les métriques de génération

**Faithfulness** (fidélité) : chaque affirmation de la réponse est-elle appuyée par le contexte fourni ? C'est la métrique anti-hallucination, et généralement la plus critique en production.

**Answer relevance** : la réponse traite-t-elle réellement la question posée ? Un texte parfaitement fidèle au contexte mais hors sujet reste inutile.

**Context precision** : les passages réellement utilisés étaient-ils bien classés en tête ?

Ces métriques se calculent le plus souvent avec un **LLM-as-a-judge** : un second modèle note la sortie selon une grille. Approche efficace mais imparfaite — le juge a ses propres biais, et il faut vérifier son accord avec une notation humaine sur un échantillon.

## Frameworks

**Ragas** est spécialisé RAG et implémente directement les métriques ci-dessus.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
)
print(results)
```

**DeepEval** adopte une approche « pytest » : les évaluations s'écrivent comme des tests et s'intègrent naturellement dans une CI, avec échec du build si un score passe sous un seuil.

## Construire son jeu de test

C'est l'étape que tout le monde saute, et c'est celle qui a le plus de valeur.

Commencez petit : **30 à 50 questions suffisent** pour détecter les régressions grossières. Chaque entrée contient la question, la réponse attendue, et le passage source qui la contient.

Trois précautions. Couvrez la diversité réelle des usages : questions factuelles simples, questions nécessitant plusieurs documents, et surtout questions **hors périmètre**, dont la bonne réponse est « je ne sais pas ». Ce dernier cas est souvent le plus révélateur.

Faites relire le jeu par un expert métier — un jeu de test généré par un LLM et jamais vérifié mesure surtout la cohérence du LLM avec lui-même.

Enfin, versionnez ce jeu avec le code et rejouez-le à chaque modification du prompt, du modèle ou de la stratégie de chunking. C'est votre seul filet de sécurité contre les régressions silencieuses.
