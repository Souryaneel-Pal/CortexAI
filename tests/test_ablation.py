from src.eval.ablation import build_ablation_report


def test_fusion_beats_every_modality_true_case():
    metrics = {
        "face_only": {"macro_f1": 0.55, "weighted_f1": 0.58, "rmse_mean": 6.2},
        "speech_only": {"macro_f1": 0.50, "weighted_f1": 0.53, "rmse_mean": 6.8},
        "tabular_only": {"macro_f1": 0.70, "weighted_f1": 0.72, "rmse_mean": 4.5},
        "fusion": {"macro_f1": 0.78, "weighted_f1": 0.79, "rmse_mean": 3.9},
    }
    report = build_ablation_report(metrics)
    assert report.fusion_beats_every_modality() is True
    assert len(report.entries) == 4


def test_fusion_beats_every_modality_false_when_a_baseline_wins():
    metrics = {
        "tabular_only": {"macro_f1": 0.85, "weighted_f1": 0.85, "rmse_mean": 3.0},  # beats fusion
        "fusion": {"macro_f1": 0.78, "weighted_f1": 0.79, "rmse_mean": 3.9},
    }
    report = build_ablation_report(metrics)
    assert report.fusion_beats_every_modality() is False


def test_fusion_beats_every_modality_false_without_fusion_entry():
    metrics = {"tabular_only": {"macro_f1": 0.7, "weighted_f1": 0.72, "rmse_mean": 4.5}}
    report = build_ablation_report(metrics)
    assert report.fusion_beats_every_modality() is False


def test_markdown_table_contains_all_sources():
    metrics = {
        "tabular_only": {"macro_f1": 0.7, "weighted_f1": 0.72, "rmse_mean": 4.5},
        "fusion": {"macro_f1": 0.78, "weighted_f1": 0.79, "rmse_mean": 3.9},
    }
    report = build_ablation_report(metrics)
    table = report.to_markdown_table()
    assert "tabular_only" in table
    assert "fusion" in table
    assert "0.780" in table


def test_save_and_load_roundtrip(tmp_path):
    metrics = {
        "tabular_only": {"macro_f1": 0.7, "weighted_f1": 0.72, "rmse_mean": 4.5},
        "fusion": {"macro_f1": 0.78, "weighted_f1": 0.79, "rmse_mean": 3.9},
    }
    report = build_ablation_report(metrics)
    path = tmp_path / "ablation.json"
    report.save(path)

    loaded = report.load(path)
    assert loaded.fusion_beats_every_modality() is True
    assert len(loaded.entries) == 2
