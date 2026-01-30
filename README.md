# Projet Glacier

Ce dépôt est l’espace de travail pour notre projet du cours **IFT3710 — Projet en apprentissage automatique**.

Pour rerun le code et reproduire les notebooks, suivez les étapes ci-dessous.

## Attention (Earthdata requis)

Pour télécharger le dataset NSIDC/GLIMS, il faut :
1) Créer un compte **NASA Earthdata**  
2) Créer un fichier **`_netrc`** à la racine du repo

Contenu du fichier `_netrc` :
```txt
machine urs.earthdata.nasa.gov
  login ...
  password ...
```
# Installation rapide
### 1) Cloner le repo
```bash
git clone https://github.com/pierre-emery/projet_glacier.git
cd projet_glacier
```
### 2) Créer un environnement virtuel (venv)
```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```
### 3) Installer les dépendances
```bash
pip install -U pip
pip install -r requirements.txt
```
### 4) Ouvrir Jupyter
```bash
jupyter notebook
# ou
jupyter lab
```
