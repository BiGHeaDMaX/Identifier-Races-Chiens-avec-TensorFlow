# **Introduction**
Ce projet vise à élaborer un modèle qui permet de prédire la race d'un chien présent sur une image.<br>
Les images utilisées pour l'entraînement proviennent du [*Stanford Dogs Dataset*](http://vision.stanford.edu/aditya86/ImageNetDogs/).

# **Contenu de ce repository**
- **01 - Analyse et prétraitements.ipynb** : rapide analyse exploratoire du dataset et prétraitements des images.
- **02 - Modèle personnel.ipynb** : élaboration et entraînement complet d'un réseau de neurones.
- **03 - Transfert learning.ipynb** : utilisation de modèles préentraînés dont nous réentraînons les dernières couches pour adapter ces derniers à notre problématique.
- **04 - Application.py** : application Streamlit avec le meilleur modèle pour faire des prédictions sur de nouvelles photos.
- **05 - Présentation.pptx** : support de présentation.

> **Note**<br>
<i>02 - Modèle personnel.ipynb</i> et <i>03 - Transfert learning.ipynb</i> ont été conçus pour fonctionner avec une carte graphique nVidia.

# **Utiliser une carte graphique pour accélérer les calculs avec Tensorflow sous Windows**
Il n'est plus possible d'utiliser directement sa carte graphique avec Tensorflow sous Windows. Il va falloir lancer un système Linux, pour ce faire, nous allons utiliser WSL (Windows Subsystem for Linux).<br>

## **Installer et lancer une distribution Linux**
- Sous Windows, lancer une invite de commande et taper : ```wsl --install```
<i>Suivre les instructions, indiquer un nom d'utilisateur (il peut être différent de celui sur Windows)</i>
- Lancer le système d'exploitation installé : ```wsl -d Ubuntu```

## **Installation des pilotes, des bibliothèques et mise en place**
- 

