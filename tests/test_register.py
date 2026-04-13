from unittest.mock import MagicMock

from kortex import register


class TestRegister:
    def test_registers_memory_provider(self):
        ctx = MagicMock()
        register(ctx)
        assert ctx.register_memory_provider.called

    def test_registers_context_engine_when_supported(self):
        ctx = MagicMock()
        register(ctx)
        assert ctx.register_context_engine.called
