import pytest
from unittest.mock import Mock
from nn.hooks import (
    Hook,
)  # Replace 'your_module' with the actual module name where Hook class is defined


# Test the Hook class
class TestHook:
    @pytest.fixture
    def mock_hook(self):
        # Create a mock instance of the Hook class
        return Mock(spec=Hook)

    def test_before_epoch(self, mock_hook):
        # Call the before_epoch method with mock arguments
        mock_hook.before_epoch(model=Mock(), data_loader=Mock(), epoch=1)
        # Assert the method is called without errors
        assert mock_hook.before_epoch.called

    def test_after_epoch(self, mock_hook):
        # Call the after_epoch method with mock arguments
        mock_hook.after_epoch(
            model=Mock(), data_loader=Mock(), epoch=1, metrics={"loss": 0.5}
        )
        # Assert the method is called without errors
        assert mock_hook.after_epoch.called

    def test_after_step(self, mock_hook):
        # Call the after_step method with mock arguments
        mock_hook.after_step(model=Mock(), data_loader=Mock(), epoch=1, step_loss=0.5)
        # Assert the method is called without errors
        assert mock_hook.after_step.called

    def test_before_step(self, mock_hook):
        # Call the before_step method with mock arguments
        mock_hook.before_step(model=Mock(), data_loader=Mock(), epoch=1)
        # Assert the method is called without errors
        assert mock_hook.before_step.called
