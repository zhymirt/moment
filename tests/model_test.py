from momentfm import MOMENTPipeline

def test_model_creation():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': 192,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True, # Freeze the patch embedding layer
            'freeze_embedder': True, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        },
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )