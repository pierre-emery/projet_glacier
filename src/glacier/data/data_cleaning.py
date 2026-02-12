from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd


def keep_outlines(gdf: gpd.GeoDataFrame, value: str = "glac_bound") -> gpd.GeoDataFrame:
    if "line_type" not in gdf.columns:
        return gdf
    return gdf[gdf["line_type"] == value]


def drop_empty_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "geometry" not in gdf.columns:
        return gdf
    gdf = gdf[gdf.geometry.notnull()]
    return gdf[~gdf.geometry.is_empty]


def parse_anlys_time(gdf: gpd.GeoDataFrame, col: str = "anlys_time") -> gpd.GeoDataFrame:
    if col not in gdf.columns:
        return gdf
    gdf = gdf.copy()
    gdf[col] = pd.to_datetime(gdf[col], errors="coerce", utc=True)
    return gdf


def parse_src_date(gdf: gpd.GeoDataFrame, col: str = "src_date") -> gpd.GeoDataFrame:
    if col not in gdf.columns:
        return gdf
    gdf = gdf.copy()
    gdf["src_date_dt"] = pd.to_datetime(gdf[col], errors="coerce", utc=True)
    return gdf


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 4326) -> gpd.GeoDataFrame:
    if getattr(gdf, "crs", None) is None:
        return gdf.set_crs(epsg)
    return gdf


def fix_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)
    return drop_empty_geometries(gdf)


def explode_multipolygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.explode(index_parts=False)


def filter_positive_area(gdf: gpd.GeoDataFrame, col: str = "area") -> gpd.GeoDataFrame:
    if col not in gdf.columns:
        return gdf
    gdf = gdf[gdf[col].notna()]
    return gdf[gdf[col] > 0]


def clean_elevations_soft(
    gdf: gpd.GeoDataFrame,
    cols: tuple[str, str, str] = ("min_elev", "mean_elev", "max_elev"),
    sentinel: float = -9999,
    set_nonpositive_to_nan: bool = False,
    enforce_order: bool = False,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    present = [c for c in cols if c in gdf.columns]
    if not present:
        return gdf

    for c in present:
        gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
        gdf.loc[gdf[c] == sentinel, c] = np.nan
        if set_nonpositive_to_nan:
            gdf.loc[gdf[c] <= 0, c] = np.nan

    if enforce_order and all(c in gdf.columns for c in cols):
        ok = (
            gdf["min_elev"].isna() | gdf["mean_elev"].isna() | gdf["max_elev"].isna()
        ) | (
            (gdf["min_elev"] <= gdf["mean_elev"]) & (gdf["mean_elev"] <= gdf["max_elev"])
        )
        gdf.loc[~ok, list(cols)] = np.nan

    return gdf


def cast_categories(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = ["primeclass", "surge_type", "term_type", "gtng_o1reg", "gtng_o2reg", "rgi_gl_typ", "conn_lvl"]
    for c in cols:
        if c in gdf.columns:
            gdf[c] = gdf[c].astype("category")
    return gdf


def drop_exact_dupes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "glac_id" not in gdf.columns or "geometry" not in gdf.columns:
        return gdf

    gdf = gdf.copy()
    gdf["_geom_wkb"] = gdf.geometry.to_wkb()

    subset = ["glac_id"] + (["src_date_dt"] if "src_date_dt" in gdf.columns else []) + ["_geom_wkb"]
    gdf = gdf.drop_duplicates(subset=subset)

    return gdf.drop(columns=["_geom_wkb"])


def clean_glims_outlines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    gdf = keep_outlines(gdf)
    gdf = drop_empty_geometries(gdf)
    gdf = ensure_crs(gdf, epsg=4326)
    gdf = fix_invalid_geometries(gdf)

    gdf = parse_src_date(gdf, col="src_date")
    gdf = parse_anlys_time(gdf, col="anlys_time")

    gdf = clean_elevations_soft(
        gdf,
        sentinel=-9999,
        set_nonpositive_to_nan=False,
        enforce_order=False,
    )

    gdf = filter_positive_area(gdf, col="area")
    gdf = cast_categories(gdf)
    gdf = drop_exact_dupes(gdf)

    return gdf.reset_index(drop=True)


def make_temporal_view(
    gdf_base: gpd.GeoDataFrame,
    min_date: str = "1900-01-01",
    max_date: str = "2026-12-31",
) -> gpd.GeoDataFrame:
    if "src_date_dt" not in gdf_base.columns:
        raise ValueError("src_date_dt missing; call clean_glims_outlines first.")

    g = gdf_base.copy()
    g = g.dropna(subset=["glac_id", "src_date_dt"])
    g = g[g["src_date_dt"].between(pd.Timestamp(min_date, tz="UTC"), pd.Timestamp(max_date, tz="UTC"))]

    out = g.dissolve(by=["glac_id", "src_date_dt"], aggfunc={"area": "median"}).reset_index()
    return out


def make_prediction_view(
    gdf_base: gpd.GeoDataFrame,
    k: int = 3,
    min_date: str = "1900-01-01",
    max_date: str = "2026-12-31",
) -> gpd.GeoDataFrame:
    temporal = make_temporal_view(gdf_base, min_date=min_date, max_date=max_date)
    counts = temporal.groupby("glac_id")["src_date_dt"].nunique()
    keep_ids = counts[counts >= k].index
    return temporal[temporal["glac_id"].isin(keep_ids)].copy()


def make_polygon_view(gdf_base: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return explode_multipolygons(gdf_base).reset_index(drop=True)