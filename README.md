# Installation et Lancement

## 1. Installation des dépendances

### Option A : Avec `uv` (Recommandé)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option B : Avec `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Lancement du projet

### Lancement standard (Explorateur de fichiers)

```bash
python main.py
```

### Lancement en spécifiant les 8 dossiers de caméras

```bash
python main.py Data/1_partie_0429_003-Camera1_M11139 Data/1_partie_0429_003-Camera2_M11140 Data/1_partie_0429_003-Camera3_M11141 Data/1_partie_0429_003-Camera4_M11458 Data/1_partie_0429_003-Camera5_M11459 Data/1_partie_0429_003-Camera6_M11461 Data/1_partie_0429_003-Camera7_M11462 Data/1_partie_0429_003-Camera8_M11463
```

Ou plus simplement avec un seul chemin et du globbing :

```bash
python main.py Data/1_partie_0429_003*
```


