# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
from OMEN import *

def run_experiment(
    city: str,
    all_modality_paths: Dict[str, str],
    labels_csv: str,
    task_names: Sequence[str],
    train_modalities: Sequence[str],
    test_modalities: Sequence[str],
    label_key: str = "GEOID",
    n_anchors: int = 96,
    tau: float = 1.0,
    proj_dim: int = 128,
    dropout: float = 0.0,
    gnn_layers: int = 2,
    gnn_dropout: float = 0.1,
    graph_k_mid: int = 16,
    graph_k_min: int = 4,
    graph_k_max: int = 64,
    anchor_reweight_init: str = "identity",
    consensus_hidden: int = 128,
    consensus_temperature: float = 1.0,
    ssl_epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 60,
    min_view_modalities: int = 1,
    inv_weight: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    val_n_views: int = 16,
    val_seed: int = 1234,
    kfold: int = 5,
    seed: int = 42,
    device_str: str = "cuda:0",
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints",
):
    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger = setup_logging(log_dir, f"OMEN_{city}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_modalities = list(train_modalities)
    test_modalities  = list(test_modalities)
    new_mods         = [m for m in test_modalities if m not in train_modalities]
    new_modality     = new_mods if len(new_mods) > 0 else None

    logger.info(f"Initial modalities : {train_modalities}")
    logger.info(f"All  modalities : {test_modalities}")
    logger.info(f"Emerging modality     : {new_modality}")

    train_modality_paths = {m: all_modality_paths[m] for m in train_modalities}
    anchors, region_ids  = select_anchors_multimodal(train_modality_paths, n_anchors, seed)
    all_relative_repr    = compute_relative_representations(all_modality_paths, anchors, region_ids, tau)
    all_repr_tensors     = {m: torch.tensor(v, dtype=torch.float32, device=device) for m, v in all_relative_repr.items()}
    train_repr = {m: all_repr_tensors[m] for m in train_modalities}
    test_repr  = {m: all_repr_tensors[m] for m in test_modalities}
    K = next(iter(all_relative_repr.values())).shape[1]
    N = len(region_ids)
    logger.info(f"Regions: {N} | Anchors: {len(anchors)} | K={K}")

    idx_all = np.arange(N)
    ssl_train_idx, ssl_val_idx = train_test_split(idx_all, test_size=0.2, random_state=seed, shuffle=True)
    train_loader = DataLoader(Subset(IndexDataset(N), ssl_train_idx), batch_size=batch_size, shuffle=True, drop_last=False)

    model = OMENPretrainingWrapper(
        K, proj_dim=proj_dim, dropout=dropout,
        gnn_layers=gnn_layers, gnn_dropout=gnn_dropout,
        graph_k_mid=graph_k_mid, graph_k_min=graph_k_min, graph_k_max=graph_k_max,
        anchor_reweight_init=anchor_reweight_init,
        consensus_hidden=consensus_hidden,
        consensus_temperature=consensus_temperature,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ssl_epochs, eta_min=lr * 0.02)
    ckpt_path = os.path.join(checkpoint_dir, f"OMEN_{city}.pt")

    val_view_pairs = build_fixed_val_views(
        train_repr, train_modalities, n_pairs=val_n_views,
        min_view_modalities=min_view_modalities, seed=val_seed,
    )

    min_val_loss = float("inf")
    bad_count = 0

    for epoch in range(1, ssl_epochs + 1):
        tr = train_ssl_epoch(
            model, train_loader, optimizer, train_repr, train_modalities, device,
            min_view_modalities=min_view_modalities,
            inv_weight=inv_weight, var_weight=var_weight, cov_weight=cov_weight,
        )
        va = validate_ssl_fixed(
            model, val_view_pairs, ssl_val_idx, device,
            inv_weight=inv_weight, var_weight=var_weight, cov_weight=cov_weight,
        )
        scheduler.step()

        if va["loss"] < min_val_loss - 1e-5 or epoch == 1:
            min_val_loss = va["loss"]
            bad_count = 0
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, min_val_loss)
        else:
            bad_count += 1

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | train_loss={tr['loss']:.4f} ")

        if epoch >= 50 and bad_count >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    load_checkpoint(ckpt_path, model, device=device)
    logger.info("Loaded SSL checkpoint.")

    z3 = extract_features(model, train_repr, device)
    z4 = extract_features(model, test_repr,  device)

    labels        = RegionLabelTable(labels_csv, task_names, region_ids, label_key=label_key)
    metrics_z3, _ = evaluate_downstream_ridge(z3, labels, task_names, kfold=kfold, seed=seed)
    metrics_z4, _ = evaluate_downstream_ridge(z4, labels, task_names, kfold=kfold, seed=seed)
    summarize_probe_results(metrics_z3, metrics_z4, task_names, logger,
                            title="Frozen encoder downstream probe: Initial vs Extended")

    logger.info("\nDone.")
    return metrics_z3, metrics_z4


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CITY = "nyc"

    NYC_ALL_TASK_NAMES = [
        "median_household_income", "bachelors_or_higher_pct", "public_transport_ratio",
        "obesity_rate", "ndvi", "population_density", "building_height_mean",
        "crime_rate", "median_gross_rent", "poverty_rate", "unemployment_rate",
        "drive_ratio", "carpool_ratio", "walk_bike_ratio", "work_from_home_ratio",
    ]
    LONDON_ALL_TASK_NAMES = [
        "median_household_income", "bachelors_or_higher_pct", "public_transport_ratio",
        "life_expectancy", "healthcare_accessibility", "ndvi_mean",
        "population_density_km2", "building_height_mean", "crime_rate",
        "poverty_rate", "unemployment_rate", "drive_ratio", "carpool_ratio",
        "walk_bike_ratio", "work_from_home_ratio",
    ]

    TASK_NAMES = NYC_ALL_TASK_NAMES if CITY.lower() == "nyc" else LONDON_ALL_TASK_NAMES
    LABEL_KEY  = "GEOID" if CITY.lower() == "nyc" else "MSOA21CD"

    ALL_MODALITY_PATHS = {
        "satellite": f"./encode_emb_{CITY}/emb_satellite.pkl",
        "poi":       f"./encode_emb_{CITY}/emb_poi.pkl",
        "street":    f"./encode_emb_{CITY}/emb_street.pkl",
        "osm":       f"./encode_emb_{CITY}/emb_osm.pkl",
    }

    run_experiment(
        city=CITY,
        all_modality_paths=ALL_MODALITY_PATHS,
        labels_csv=f"./data/{CITY}/labels.csv",
        task_names=TASK_NAMES,
        train_modalities=["street", "satellite",],
        test_modalities=["street", "osm", "satellite", "poi"],
        label_key=LABEL_KEY,
        n_anchors=768,
        tau=1.0,
        proj_dim=128,
        dropout=0.0,
        gnn_layers=2,
        gnn_dropout=0.1,
        graph_k_mid=16,
        graph_k_min=4,
        graph_k_max=64,
        consensus_hidden=128,
        consensus_temperature=1.0,
        ssl_epochs=400,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        patience=60,
        min_view_modalities=1,
        inv_weight=1.0,
        var_weight=1.0,
        cov_weight=0.04,
        val_n_views=16,
        val_seed=42,
        kfold=5,
        seed=42,
        device_str="cuda:0",
        log_dir="./logs_OMEN",
        checkpoint_dir="./checkpoints_OMEN",
    )