# Introduction à Docker pour la Data Science

## Pourquoi containeriser ?

« Ça marche sur ma machine. » Cette phrase est probablement la plus coûteuse de l'histoire du développement logiciel. En data science, le problème est amplifié : votre projet dépend d'une version précise de Python, de scikit-learn, parfois de CUDA, et d'une bibliothèque système comme `libgomp`. Reproduire cet environnement sur le serveur de production relève souvent de l'archéologie.

Docker répond à ce problème en empaquetant votre code **et** son environnement d'exécution dans une unité unique et portable. Le même artefact tourne sur votre laptop, sur la CI, et en production.

## Les trois concepts de base

**L'image** est un modèle en lecture seule : un système de fichiers figé contenant Python, vos dépendances et votre code. On la construit une fois, à partir d'un `Dockerfile`, puis on la partage via un registry (Docker Hub, ECR, GitLab Registry).

**Le container** est une instance en cours d'exécution d'une image. On peut en lancer dix à partir de la même image ; chacun a son propre système de fichiers temporaire. Point crucial : **tout ce qu'un container écrit disparaît quand il est supprimé**.

**Le volume** résout ce dernier point. C'est un espace de stockage géré par Docker et monté dans le container, qui survit à sa destruction. C'est là qu'on met une base de données, des modèles entraînés, ou un index vectoriel qu'on ne veut pas reconstruire à chaque redémarrage.

## Un premier Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/train.py"]
```

Deux détails valent une explication.

D'abord, l'image `slim` fait environ 130 Mo contre 1 Go pour l'image complète. Moins de packages installés, c'est aussi moins de surface d'attaque.

Ensuite, l'ordre des instructions n'est pas arbitraire. Docker met en cache chaque étape. En copiant `requirements.txt` **avant** le code source, on garantit que modifier un fichier Python ne réinvalide pas l'étape `pip install`. Sur un projet ML avec PyTorch, c'est la différence entre un build de 5 secondes et un build de 4 minutes.

## Construire et lancer

```bash
docker build -t mon-projet-ml:0.1 .
docker run --rm -v $(pwd)/data:/app/data mon-projet-ml:0.1
```

L'option `-v` monte le dossier `data/` local dans le container : le script y lit ses données d'entrée et y écrit son modèle, et le résultat reste sur votre machine après l'arrêt.

## À retenir

Un `.dockerignore` évite de copier `.git/`, `.env` et `venv/` dans l'image — sinon on embarque des secrets et plusieurs centaines de mégaoctets inutiles. Et on épingle toujours les versions dans `requirements.txt` : sans cela, deux builds à une semaine d'intervalle produisent deux environnements différents, ce qui annule tout le bénéfice de la containerisation.
