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
<i>02 - Modèle personnel.ipynb</i> et <i>03 - Transfert learning.ipynb</i> ont été conçus pour fonctionner avec une carte graphique nVidia. Ils peuvent fonctionner sans carte graphique, mais leur exécution sera beaucoup plus longue.

# **Article associé**
Retrouvez l'article de présentation de ce projet [ici](https://bigheadmax.github.io/05-classer-des-images.html).

# **Utiliser une carte graphique pour accélérer les calculs avec Tensorflow sous Windows**
Il n'est plus possible d'utiliser directement sa carte graphique avec Tensorflow sous Windows. Il va falloir lancer un système Linux, pour ce faire, nous allons utiliser WSL (Windows Subsystem for Linux).<br>
Le plus souvent, les problèmes proviennent d'une incompatibilité entre Tensorflow et les drivers nVidia, il ne faut donc pas toujours utiliser les dernières versions.

## **Installer et lancer une distribution Linux**
- Sous Windows, lancer une invite de commande et taper : ```wsl --install```<br>
<i>Suivre les instructions, indiquer un nom d'utilisateur (il peut être différent de celui sur Windows)</i>
- Lancer le système d'exploitation installé : ```wsl -d Ubuntu```<br>
<i>Préciser "Ubuntu" permet de lancer le bon système, car plusieurs peuvent coexister (par exemple Docker)</i>

## **Installation des pilotes, des bibliothèques et mise en place**
- Une fois sous Linux, entrer les commandes suivantes : <br>
```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```
- Les chemins vers certains fichiers doivent être précisés : <br>
```
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.zshrc
sudo ln -s /usr/lib/wsl/lib/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so
```
- Aller sur [ce site](https://developer.nvidia.com/rdp/cudnn-download) (il faut créer un compte, c'est rapide et gratuit) et télécharger <i>cuDNN</i> pour <i>CUDA 11.x</i> pour Linux.<br>
- Mettre le fichier compressé dans le dossier dans lequel on se trouve. Le nom du fichier <i>.tar.xz</i> peut varier, adapter en conséquence et entrer les commandes suivantes : <br>
```
tar -xvf cudnn-linux-x86_64-8.9.4.25_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.9.4.25_cuda11-archive
sudo cp -P include/* /usr/local/cuda/include/
sudo cp -P lib/* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig /usr/local/cuda/lib64
```
- Installer Python, Tensorflow et d'autres bibliothèques si besoin : <br>
```
sudo apt update && sudo apt upgrade
sudo apt install python3-pip
pip install --upgrade pip
pip install --upgrade pip setuptools wheel
pip install tensorflow
```
Si besoin (carte graphique non détectée lors de l'exécution de votre code) : <br>
```pip install tensorflow[and-cuda]```

## **Quelques commandes utiles**
- Accéder aux dossiers de votre système Linux depuis l'explorateur Window : <br>
```\\wsl.localhost\Ubuntu\home\Nom_Utilisateur```
- Afficher tous les fichiers/dossiers (y compris cachés avec -la) : <br>
```ls -la```
- Accéder aux fichiers Windows depuis Ubuntu WSL, revenir à la racine si besoin (```cd ..```), puis : <br>
```cd mnt/c```
- Afficher où on se trouve actuellement : <br>
```pwd```
- Lancer un fichier python depuis Powershell (par exemple), en utilisant WSL : <br>
```wsl -d Ubuntu python3 hello.py```
- Supprimer Ubuntu : <br>
```wsl --unregister Ubuntu```<br>
<i>Et désinstaller si besoin Ubuntu depuis la liste des logiciels installés (Windows)</i>

## **Exécuter un notebook sur WSL depuis Visual Studio Code**
- Tout en bas à gauche, cliquer sur le bouton bleu (<i>Open a remote Windows</i>)<br>
- Cliquer ensuite sur <i>Connect to WSL using Distro...</i><br>
- Choisir <i>Ubuntu</i><br>
- Pour ouvrir vos fichiers qui se trouvent sur votre Windows, remonter deux fois ```..```, ```..``` puis ```mnt```, ```c``` et enfin ```Users``` (par exemple)<br>
> **Note**<br>
Puisque vous serez sur un nouveau système et non plus sous Windows, il vous sera nécessaire de réinstaller d'éventuelles extensions de VS Code (comme Jupyter, GitHub, etc).
- Pour ne plus exécuter votre code sur Linux et retourner sous Window, cliquer de nouveau en bas à gauche, puis sur <i>Close Remote Connection</i>