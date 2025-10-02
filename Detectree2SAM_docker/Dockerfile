FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copier les scripts dans le conteneur
COPY detectree2SAM_main.py .
COPY detectree2SAM_lib.py .

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer dépendances système nécessaires
RUN echo "🟢 Étape 1 : Mise à jour apt-get et installation des dépendances système (git, build-essential, gdal, libgl1)" && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential gdal-bin libgdal-dev libgl1 tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip, wheel, setuptools
RUN echo "🟢 Étape 2 : Mise à jour pip, wheel et setuptools" && \
    pip install --upgrade pip wheel setuptools

# Installer toutes les dépendances Python via requirements.txt
RUN echo "🟢 Étape 3 : Installation des dépendances Python depuis requirements.txt" && \
    pip install -r requirements.txt

CMD ["python", "detectree2SAM_main.py"]
