import pytest
from unittest.mock import patch
import torch
import torch.optim as optim
from nn.basemodel import BaseModel


# ConcreteModel is a subclass of BaseModel with a single linear layer.
# It's used to provide a concrete implementation for testing the abstract BaseModel class.
class ConcreteModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Adding a simple linear layer with parameters
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Test the save_model method of the BaseModel.
def test_save_model():
    model = ConcreteModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    save_path = "test_model.pth"

    # Mock the 'torch.save' function to avoid actual file I/O during test
    with patch("torch.save") as mock_save:
        # Save the model and check if 'torch.save' was called correctly
        model.save_model(
            save_path, optimizer, epoch=1, additional_info={"info": "test"}
        )
        assert mock_save.called
        # Extract the saved dictionary and verify its contents
        save_dict = mock_save.call_args[0][0]
        assert "model_state_dict" in save_dict
        assert "optimizer_state_dict" in save_dict
        assert save_dict["epoch"] == 1
        assert save_dict["info"] == "test"


# Test the load_model method of the BaseModel.
def test_load_model():
    model = ConcreteModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    load_path = "test_model.pth"

    # Create mock states for model and optimizer
    mock_model_state = {
        "linear.weight": torch.tensor([[1.0]]),
        "linear.bias": torch.tensor([0.0]),
    }

    # Get parameter IDs of the model to mock the optimizer state
    param_ids = list(map(id, model.parameters()))

    # Mock optimizer state to match the expected structure
    mock_optimizer_state = {
        "state": {},
        "param_groups": [
            {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "params": param_ids,  # Include parameter IDs here
            }
        ],
    }

    # Mock 'torch.load' to return the created mock states
    with patch(
        "torch.load",
        return_value={
            "model_state_dict": mock_model_state,
            "optimizer_state_dict": mock_optimizer_state,
        },
    ):
        # Load the model and verify if the loaded states match the mock states
        checkpoint = model.load_model(load_path, optimizer)
        assert checkpoint["model_state_dict"] == mock_model_state
        assert optimizer.state_dict()["param_groups"][0]["lr"] == 0.001
