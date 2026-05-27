from src.data_loader import load_data


def test_load_data():

    df = load_data(
        "tests/data/sample_insurance_data.txt"
    )

    assert not df.empty

    assert "LossRatio" in df.columns

    assert "VehicleAge" in df.columns

    assert df.shape[0] == 3