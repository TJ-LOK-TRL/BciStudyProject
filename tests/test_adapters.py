from src.data_adapters.motor_imagery_data_adapter import MotorImageryDataAdapter

def test_data_adapter_common_shape():
    cfg = {"window_size": 500}
    ad = MotorImageryDataAdapter(root="data/raw/motor_imagery", config=cfg)
    batch = ad.to_common(segmented=None)
    assert batch.X.shape[1] == 32
    assert batch.X.shape[2] == 500
    assert len(batch.meta) == batch.X.shape[0]
