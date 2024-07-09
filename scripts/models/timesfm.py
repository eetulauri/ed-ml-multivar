import timesfm


def get_model(y_train, y_val, pc_train, fc_train, pc_val, fc_val, optimize, **kwargs):
    """
    """
    model = timesfm.TimesFm(
        context_len=512,
        horizon_len=24,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend='gpu',
    )
    model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    study = None
    return model, study