def add_glims_history_features(temporal_all: gpd.GeoDataFrame, temporal_targets: gpd.GeoDataFrame) -> pd.DataFrame:
    all_df = temporal_all[["glac_id", "src_date_dt", "area"]].dropna(subset=["glac_id", "src_date_dt"]).copy()
    tgt_df = temporal_targets[["glac_id", "src_date_dt"]].dropna().copy()

    all_df["year"] = all_df["src_date_dt"].dt.year + all_df["src_date_dt"].dt.dayofyear / 365.25

    all_df = all_df.sort_values(["glac_id", "src_date_dt"])
    tgt_df = tgt_df.sort_values(["glac_id", "src_date_dt"])

    feats = []
    for gid, group in all_df.groupby("glac_id", sort=False):
        g_targets = tgt_df[tgt_df["glac_id"] == gid]
        if g_targets.empty:
            continue

        years = group["year"].to_numpy()
        areas = group["area"].to_numpy()
        dates = group["src_date_dt"].to_numpy()

        for t in g_targets["src_date_dt"].to_numpy():
            mask = dates < t
            past_years = years[mask]
            past_areas = areas[mask]

            n_past = past_areas.size
            if n_past == 0:
                feats.append((gid, t, 0, np.nan, np.nan, np.nan, np.nan))
                continue

            span_years = past_years.max() - past_years.min() if n_past >= 2 else 0.0
            area_last = past_areas[-1]
            delta_last = (past_areas[-1] - past_areas[-2]) if n_past >= 2 else np.nan

            # pente linéaire (trend) si >=2 points
            if n_past >= 2 and np.isfinite(past_areas).sum() >= 2:
                slope = np.polyfit(past_years, past_areas, 1)[0]
            else:
                slope = np.nan

            feats.append((gid, t, n_past, span_years, slope, area_last, delta_last))

    feats_df = pd.DataFrame(
        feats,
        columns=["glac_id", "src_date_dt", "n_past", "span_years_past", "area_trend_per_year", "area_last", "delta_area_last"],
    )
    return feats_df