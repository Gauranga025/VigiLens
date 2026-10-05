"""
Tests for causal streaming LSTM module.
"""

import torch
from models.temporal_lstm import CausalLSTM, NextFeaturePredictor, TemporalModel


def test_lstm_forward():
    """Test LSTM forward pass."""
    lstm = CausalLSTM(input_size=512, hidden_size=256, num_layers=2, dropout=0.2)
    x = torch.randn(2, 10, 512)  # (B, T, input_size)
    output, (h, c) = lstm(x)
    assert output.shape == (2, 10, 256)
    assert h.shape == (2, 2, 256)
    assert c.shape == (2, 2, 256)


def test_lstm_single_step():
    """Test LSTM single step forward."""
    lstm = CausalLSTM(input_size=512, hidden_size=256, num_layers=2, dropout=0.2)
    x = torch.randn(2, 512)  # (B, input_size)
    x = x.unsqueeze(1)  # Add sequence dim
    output, (h, c) = lstm(x)
    assert output.shape == (2, 1, 256)


def test_lstm_hidden_state_persistence():
    """Test that hidden state persists across steps."""
    lstm = CausalLSTM(input_size=512, hidden_size=256, num_layers=2, dropout=0.2)

    # First step
    x1 = torch.randn(1, 512).unsqueeze(1)
    _, (h1, c1) = lstm(x1)

    # Second step with previous hidden state
    x2 = torch.randn(1, 512).unsqueeze(1)
    _, (h2, c2) = lstm(x2, (h1, c1))

    # Hidden states should be different (updated)
    assert not torch.allclose(h1, h2)


def test_lstm_init_hidden():
    """Test hidden state initialization."""
    lstm = CausalLSTM(input_size=512, hidden_size=256, num_layers=2, dropout=0.2)
    h, c = lstm.init_hidden(batch_size=4, device=torch.device('cpu'))
    assert h.shape == (2, 4, 256)
    assert c.shape == (2, 4, 256)
    assert torch.allclose(h, torch.zeros_like(h))
    assert torch.allclose(c, torch.zeros_like(c))


def test_predictor_forward():
    """Test prediction head forward pass."""
    predictor = NextFeaturePredictor(hidden_size=256, output_size=512)
    hidden = torch.randn(2, 256)  # Last layer hidden state
    prediction = predictor(hidden.unsqueeze(0))  # Add layer dim
    assert prediction.shape == (2, 512)


def test_temporal_model_forward():
    """Test complete temporal model forward pass."""
    model = TemporalModel(input_size=512, hidden_size=256, num_layers=2, dropout=0.2)
    x = torch.randn(2, 512).unsqueeze(1)
    prediction, (h, c) = model(x)
    assert prediction.shape == (2, 512)


def test_temporal_model_hidden_state():
    """Test temporal model hidden state persistence."""
    model = TemporalModel(input_size=512, hidden_size=256, num_layers=2, dropout=0.2)

    # First step
    x1 = torch.randn(1, 512).unsqueeze(1)
    pred1, (h1, c1) = model(x1)

    # Second step with previous hidden state
    x2 = torch.randn(1, 512).unsqueeze(1)
    pred2, (h2, c2) = model(x2, (h1, c1))

    # Predictions should be different
    assert not torch.allclose(pred1, pred2)
