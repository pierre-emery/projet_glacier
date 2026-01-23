from __future__ import annotations

from pathlib import Path
import os
import requests
from dotenv import load_dotenv

BASE_URL = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0272_GLIMS_v1/"
VERSION_TAG = "v01.0"  # si ça change un jour, vous modifiez ici

def _session() -> requests.Session:
    """ Pour télécharger le jeu de données il faut créer un compte Earthdata.
    Le truc c'est qu'on ne veut pas commit le username et mot de passe du compte sur github.
    Donc on mets dans le .gitignore un fichier projet_glacier/.env qui contient :
    EARTHDATA_USERNAME=ton_username
    EARTHDATA_PASSWORD=ton_password 
    """

    load_dotenv()
    user = os.getenv("EARTHDATA_USERNAME")
    pwd = os.getenv("EARTHDATA_PASSWORD")
    if not user or not pwd:
        raise RuntimeError(
            "Missing Earthdata credentials."
        )
    s = requests.Session()
    s.auth = (user, pwd)
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

def fetch_data(date_yyyymmdd: str, raw_dir: str | Path = "data/raw/glims_v1") -> list[Path]:
    """
    Fetch GLIMS NSIDC-0272 pour une date spécifique.
    Downloads north+south zip + md5 into raw_dir.
    Le md5 sert à vérifier que le zip n'est pas corrompu.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    s = _session()
    targets = _targets_for_date(date_yyyymmdd)

    downloaded: list[Path] = []
    for name in targets:
        out = raw_dir / name
        _download_one(s, BASE_URL + name, out)
        downloaded.append(out)

    return downloaded