from __future__ import annotations

from pathlib import Path
import os
import requests
from dotenv import load_dotenv
import zipfile


BASE_URL = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0272_GLIMS_v1/"
VERSION_TAG = "v01.0"  # si ça change un jour, vous modifiez ici

def _session() -> requests.Session:
    """ Pour télécharger le jeu de données il faut créer un compte Earthdata.
    Le truc c'est qu'on ne veut pas commit le username et mot de passe du compte sur github.
    Donc on mets dans le .gitignore un fichier projet_glacier/_netrc qui contient:
    machine urs.earthdata.nasa.gov
     login ...
     password ...
    """
    root = repo_root()
    netrc_path = root / "_netrc"
    os.environ["NETRC"] = str(netrc_path)  

    s = requests.Session()
    s.trust_env = True
    return s

def _targets_for_date(date_yyyymmdd: str) -> list[str]:
    """
    Download les 4 fichiers (north/south + md5).
    date_yyyymmdd: "20260114"
    """
    if not (isinstance(date_yyyymmdd, str) and date_yyyymmdd.isdigit() and len(date_yyyymmdd) == 8):
        raise ValueError("date_yyyymmdd must be a string like '20260114'")

    def zip_name(region: str) -> str:
        return f"NSIDC-0272_glims_db_{region}_{date_yyyymmdd}_{VERSION_TAG}.zip"

    north = zip_name("north")
    south = zip_name("south")
    return [north, north + ".md5", south, south + ".md5"]

def _download_one(session: requests.Session, url: str, out_path: Path) -> None:
    """
    Atomic download: writes to .part then renames, to avoid partial/corrupt files.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    # skip si déjà downloaded
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp.replace(out_path)

def repo_root(start: Path | None = None) -> Path:
    """
    Helper pour s'assurer que l'on mets les données dans le dossier data au lieu de dans notebooks.
    """
    p = (start or Path(__file__).resolve()).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Repo root not found (no .git).")

def fetch_data(date_yyyymmdd: str, raw_dir: str | Path = "data/raw/glims_v1") -> list[Path]:
    """
    Fetch GLIMS NSIDC-0272 pour une date spécifique.
    Télécharge north+south zip + md5 dans raw_dir.
    Le md5 sert à vérifier que le zip n'est pas corrompu.
    """
    root = repo_root()

    raw_dir = Path(raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    s = _session()
    targets = _targets_for_date(date_yyyymmdd)

    downloaded: list[Path] = []
    for name in targets:
        out = raw_dir / name
        _download_one(s, BASE_URL + name, out)
        downloaded.append(out)

    return downloaded

def unzip_to(paths: list[Path], extracted_root: Path) -> list[Path]:
    """
    Dézippe les fichiers .zip dans extracted_root.
    Skip si le dossier d'extraction existe déjà et n'est pas vide.
    Ignore les fichiers non-.zip (.zip.md5 dans notre cas).
    """
    extracted_root = Path(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)

    out_dirs: list[Path] = []
    for p in paths:
        p = Path(p)

        if not p.name.lower().endswith(".zip"): # ignore les .zip.md5
            continue

        dest = extracted_root / p.stem  # stem enlève juste .zip

        # skip si déjà extrait (dossier existe et contient qqch)
        if dest.exists() and any(dest.iterdir()):
            out_dirs.append(dest)
            continue

        dest.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(dest)
        except zipfile.BadZipFile as e:
            continue
        out_dirs.append(dest)

