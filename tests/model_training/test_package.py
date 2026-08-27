def test_package_imports():
    import cfb_model_build.model_training as model_training
    assert hasattr(model_training, "__version__")
