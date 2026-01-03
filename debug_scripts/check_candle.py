try:
    import candle_core
    print("candle_core version:", candle_core.__version__)
    from candle_core import VarBuilder, DType, Device
    print("Successfully imported VarBuilder")
except ImportError:
    print("candle_core not found")
except Exception as e:
    print(f"Error: {e}")
