import pytest

from src import config, data_ingestion, model, pipeline


def test_ingestion_raises_on_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(data_ingestion.DataIngestionError):
        data_ingestion.load_appointment_data(missing_path)


def test_temporal_split_raises_on_out_of_range_date():
    import pandas as pd
    df = pd.DataFrame({"appointment_date": ["2025-01-01", "2025-01-02"]})
    with pytest.raises(pipeline.PipelineError):
        pipeline.temporal_train_test_split(df, split_date="2099-01-01")


@pytest.mark.skipif(
    not config.RAW_DATA_PATH.exists(), reason="raw dataset not available in this environment"
)
def test_full_pipeline_runs_end_to_end_on_real_data():
    """Smoke test: the full pipeline should run without error on the actual
    HealthConnect dataset and produce a trainable, evaluable baseline model."""
    result = pipeline.run_pipeline(save_outputs=False)

    assert len(result["train"]) > 0
    assert len(result["test"]) > 0
    assert "target" in result["train"].columns
    assert "waiting_time_minutes" not in result["train"].columns

    trained_model, feature_cols = model.train_baseline_model(result["train"])
    metrics = model.evaluate_model(trained_model, result["test"], feature_cols)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
