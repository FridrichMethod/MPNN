from mpnn.env import PROJECT_ROOT_DIR
from mpnn.stabddg.folding_dataset import MegascaleDataset, ThermoMutDBDataset
from mpnn.stabddg.model import StaBddG
from mpnn.stability_finetune import validation_step


def eval_pretrained_mpnn(
    pretrained_model,
    batch_size=10000,
    device="cuda",
    mc_samples=20,
    backbone_noise=0.2,
    megascale_split_path=PROJECT_ROOT_DIR / "datasets/megascale/mega_splits.pkl",
    megascale_pdb_dir=PROJECT_ROOT_DIR / "datasets/megascale/AlphaFold_model_PDBs",
    megascale_csv=PROJECT_ROOT_DIR
    / "datasets/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
    fsd_thermo_csv=PROJECT_ROOT_DIR / "datasets/FSD/fsd_thermo.csv",
    fsd_thermo_pdb_dir=PROJECT_ROOT_DIR / "datasets/FSD/PDBs",
    fsd_thermo_cache_path=PROJECT_ROOT_DIR / "datasets/FSD/fsd_thermo.pkl",
):

    model = StaBddG(
        pmpnn=pretrained_model,
        use_antithetic_variates=True,
        noise_level=backbone_noise,
        device=device,
    )

    megascale_train = MegascaleDataset(
        csv_path=megascale_csv,
        pdb_dir=megascale_pdb_dir,
        split_path=megascale_split_path,
        split="train",
    )
    megascale_valid = MegascaleDataset(
        csv_path=megascale_csv,
        pdb_dir=megascale_pdb_dir,
        split_path=megascale_split_path,
        split="val",
    )
    megascale_test = MegascaleDataset(
        csv_path=megascale_csv,
        pdb_dir=megascale_pdb_dir,
        split_path=megascale_split_path,
        split="test",
    )
    fsd_thermo_train = ThermoMutDBDataset(
        csv_path=fsd_thermo_csv,
        pdb_dir=fsd_thermo_pdb_dir,
        pdb_dict_cache_path=fsd_thermo_cache_path,
        cif=False,
    )

    train_metrics = validation_step(
        model,
        megascale_train,
        batch_size=batch_size,
        name="train",
        device=device,
        mc_samples=mc_samples,
    )
    valid_metrics = validation_step(
        model,
        megascale_valid,
        batch_size=batch_size,
        name="valid",
        device=device,
        mc_samples=mc_samples,
    )
    test_metrics = validation_step(
        model,
        megascale_test,
        batch_size=batch_size,
        name="test",
        device=device,
        mc_samples=mc_samples,
    )
    fsd_thermo_metrics = validation_step(
        model,
        fsd_thermo_train,
        batch_size=batch_size,
        name="fsd_thermo",
        device=device,
        mc_samples=mc_samples,
    )

    # convert all metric values to floating point
    train_metrics = {k: float(v) for k, v in train_metrics.items()}
    valid_metrics = {k: float(v) for k, v in valid_metrics.items()}
    test_metrics = {k: float(v) for k, v in test_metrics.items()}
    fsd_thermo_metrics = {k: -1 * float(v) for k, v in fsd_thermo_metrics.items()}
    return train_metrics, valid_metrics, test_metrics, fsd_thermo_metrics
