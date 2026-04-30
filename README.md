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
## Utilité et ordre d'exécution des notebooks

Certains notebooks produisent les données nécessaires au suivant et il faut donc les runs dans cet ordre :

### 1. `1_data_fetch_clean.ipynb` — Construction du jeu de données

Nettoie les contours GLIMS, télécharge les scènes Sentinel-2 correspondantes via STAC, applique le score de qualité pour filtrer les images, construit les composites médians et rastérise les masques de supervision.

>  Cette étape est longue (téléchargement de ~600 composites). Nécessite le fichier `_netrc` configuré (voir plus haut).

### 2. `model.ipynb` — Entraînement des U-Net

C'est avec ce notebook qu'on a entraîné les quatre variantes U-Net comparées dans le rapport (baseline, dropout+wd seul, augmentation seule, combinaison complète) sur les composites du Caucase. Sauvegarde le meilleur checkpoint (IoU validation max) pour chaque configuration.

> Compter ~7-8h par modèle sur CPU.

### 3. `deeplabv3.ipynb` — Entraînement de DeepLabV3+

Entraîne le modèle de comparaison DeepLabV3+ (ResNet50 pré-entraîné sur ImageNet) avec adaptation à 5 bandes spectrales.

> Compter ~22h sur CPU.

### 4. `test.ipynb` — Évaluation finale et figures

Charge tous les checkpoints, évalue chaque modèle sur les ensembles de test (Alpes, Andes, global), génère les tableaux de métriques.

### Notebooks de développement

Les notebooks `supervision_mask.ipynb` et `visualisation_composites.ipynb` ne
sont pas nécessaires pour reproduire les résultats : ils ont servi à concevoir
les stratégies utilisées dans `1_data_fetch_clean.ipynb`.

- **`supervision_mask.ipynb`** : exploration de la stratégie de génération des
  masques de supervision à partir des polygones GLIMS.
- **`visualisation_composites.ipynb`** : visualisation de la couche SCL et du
  score de qualité, qui a permis de mettre au point la stratégie de filtrage et
  de sélection des scènes par région.
  
## Structure du projet

```
projet_glacier/
├── data/
│   ├── raw/                # Outlines GLIMS bruts (téléchargés depuis NSIDC)
│   ├── processed/          # Outlines nettoyés (parquet)
│   ├── sentinel2/          # Composites Sentinel-2 et masques (générés)
│   └── checkpoints/        # Poids des modèles entraînés (.pt)
├── notebooks/              # Notebooks d'expérimentation et d'évaluation
│   ├── data_fetch_clean.ipynb
│   ├── model.ipynb
│   ├── deeplabv3.ipynb
│   ├── test.ipynb
│   ├── supervision_mask.ipynb
│   └── visualisation_composites.ipynb
├── reports/                # Sources LaTeX des rapports remis (cours IFT3710)
│   ├── figures/            # Figures utilisées dans les rapports
│   ├── rapport_final.tex
│   ├── revue.tex
│   ├── proposition_projet.tex
│   ├── méthodes_préliminaires.tex
│   └── references.bib
├── src/glacier/            # Modules Python réutilisables (importables)
│   ├── data/               # Nettoyage GLIMS, fetch Sentinel-2 (STAC)
│   ├── model/              # Architectures U-Net, losses, training
│   └── visualisation/      # Utilitaires de visualisation
├── requirements.txt
└── README.md
```
