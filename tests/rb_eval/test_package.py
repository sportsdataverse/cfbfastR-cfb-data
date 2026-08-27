def test_package_imports():
    import cfb_model_build.rb_eval as rb_eval
    assert hasattr(rb_eval, "__version__")
