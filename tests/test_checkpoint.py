"""Tests for checkpoint utilities."""

import hashlib
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from tianwen.utils.checkpoint import (
    compute_checkpoint_hash,
    save_checkpoint,
    verify_checkpoint_hash,
    load_checkpoint,
    load_state_dict_safe,
    get_checkpoint_info,
)
from tianwen.utils.errors import CheckpointError


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointHash:
    """Test checkpoint hash computation."""

    def test_compute_hash(self, tmp_path):
        """Test computing checkpoint hash."""
        ckpt_path = tmp_path / "test.pt"
        ckpt_path.write_bytes(b"test checkpoint data")

        hash_value = compute_checkpoint_hash(ckpt_path)

        # Hash should be 64-character hex string (SHA256)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_hash_consistency(self, tmp_path):
        """Test hash is consistent for same file."""
        ckpt_path = tmp_path / "test.pt"
        ckpt_path.write_bytes(b"test data")

        hash1 = compute_checkpoint_hash(ckpt_path)
        hash2 = compute_checkpoint_hash(ckpt_path)

        assert hash1 == hash2

    def test_hash_different_files(self, tmp_path):
        """Test different files have different hashes."""
        ckpt1 = tmp_path / "test1.pt"
        ckpt2 = tmp_path / "test2.pt"

        ckpt1.write_bytes(b"data1")
        ckpt2.write_bytes(b"data2")

        hash1 = compute_checkpoint_hash(ckpt1)
        hash2 = compute_checkpoint_hash(ckpt2)

        assert hash1 != hash2


class TestSaveCheckpoint:
    """Test checkpoint saving."""

    def test_save_simple_checkpoint(self, tmp_path):
        """Test saving a simple checkpoint."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param1": torch.randn(5), "param2": torch.randn(3, 3)}

        save_checkpoint(ckpt_path, state_dict)

        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 0

    def test_save_checkpoint_with_metadata(self, tmp_path):
        """Test saving checkpoint with metadata."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param": torch.randn(5)}
        metadata = {
            "epoch": 10,
            "best_metric": 0.95,
            "model_name": "test_model"
        }

        save_checkpoint(ckpt_path, state_dict, metadata=metadata)

        # Load and verify
        loaded = torch.load(ckpt_path)
        assert "metadata" in loaded
        assert loaded["metadata"]["epoch"] == 10
        assert loaded["metadata"]["best_metric"] == 0.95

    def test_save_checkpoint_with_hash(self, tmp_path):
        """Test saving checkpoint with hash file."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param": torch.randn(5)}

        save_checkpoint(ckpt_path, state_dict, save_hash=True)

        # Check hash file exists
        hash_path = ckpt_path.with_suffix(ckpt_path.suffix + ".sha256")
        assert hash_path.exists()

        # Verify hash matches
        saved_hash = hash_path.read_text().strip()
        actual_hash = compute_checkpoint_hash(ckpt_path)
        assert saved_hash == actual_hash

    def test_save_checkpoint_creates_dirs(self, tmp_path):
        """Test saving checkpoint creates parent directories."""
        ckpt_path = tmp_path / "subdir1" / "subdir2" / "checkpoint.pt"
        state_dict = {"param": torch.randn(5)}

        save_checkpoint(ckpt_path, state_dict)

        assert ckpt_path.exists()
        assert ckpt_path.parent.exists()


class TestVerifyCheckpointHash:
    """Test checkpoint hash verification."""

    def test_verify_hash_matches(self, tmp_path):
        """Test verification succeeds when hash matches."""
        ckpt_path = tmp_path / "checkpoint.pt"
        ckpt_path.write_bytes(b"test data")

        expected_hash = compute_checkpoint_hash(ckpt_path)
        result = verify_checkpoint_hash(ckpt_path, expected_hash=expected_hash)

        assert result is True

    def test_verify_hash_mismatch(self, tmp_path):
        """Test verification fails when hash doesn't match."""
        ckpt_path = tmp_path / "checkpoint.pt"
        ckpt_path.write_bytes(b"test data")

        wrong_hash = "0" * 64
        result = verify_checkpoint_hash(ckpt_path, expected_hash=wrong_hash)

        assert result is False

    def test_verify_hash_from_file(self, tmp_path):
        """Test verification reads hash from .sha256 file."""
        ckpt_path = tmp_path / "checkpoint.pt"
        ckpt_path.write_bytes(b"test data")

        # Create hash file
        hash_value = compute_checkpoint_hash(ckpt_path)
        hash_path = ckpt_path.with_suffix(ckpt_path.suffix + ".sha256")
        hash_path.write_text(hash_value)

        # Verify without providing hash
        result = verify_checkpoint_hash(ckpt_path)
        assert result is True


class TestLoadCheckpoint:
    """Test checkpoint loading."""

    def test_load_simple_checkpoint(self, tmp_path):
        """Test loading a simple checkpoint."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param": torch.randn(5)}
        torch.save(state_dict, ckpt_path)

        loaded = load_checkpoint(ckpt_path)

        assert "param" in loaded
        assert torch.allclose(loaded["param"], state_dict["param"])

    def test_load_checkpoint_with_verification(self, tmp_path):
        """Test loading with hash verification."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param": torch.randn(5)}

        # Save with hash
        save_checkpoint(ckpt_path, state_dict, save_hash=True)

        # Load with verification
        loaded = load_checkpoint(ckpt_path, verify_hash=True)

        assert "state_dict" in loaded

    def test_load_checkpoint_nonexistent_fails(self, tmp_path):
        """Test loading non-existent checkpoint fails."""
        ckpt_path = tmp_path / "nonexistent.pt"

        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(ckpt_path)

    def test_load_checkpoint_map_location(self, tmp_path):
        """Test loading checkpoint with map_location."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param": torch.randn(5)}
        torch.save(state_dict, ckpt_path)

        # Load to CPU
        loaded = load_checkpoint(ckpt_path, map_location="cpu")

        assert loaded is not None


class TestLoadStateDictSafe:
    """Test safe state dict loading into model."""

    def test_load_state_dict_exact_match(self, tmp_path):
        """Test loading when state dict exactly matches model."""
        model = SimpleModel()
        ckpt_path = tmp_path / "checkpoint.pt"

        # Save model state
        save_checkpoint(ckpt_path, model.state_dict())

        # Create new model and load
        new_model = SimpleModel()
        result = load_state_dict_safe(new_model, ckpt_path)

        assert "missing_keys" in result
        assert "unexpected_keys" in result
        assert len(result["missing_keys"]) == 0
        assert len(result["unexpected_keys"]) == 0

    def test_load_state_dict_with_metadata(self, tmp_path):
        """Test loading preserves metadata."""
        model = SimpleModel()
        ckpt_path = tmp_path / "checkpoint.pt"
        metadata = {"epoch": 5, "accuracy": 0.9}

        # Save with metadata
        save_checkpoint(ckpt_path, model.state_dict(), metadata=metadata)

        # Load
        new_model = SimpleModel()
        result = load_state_dict_safe(new_model, ckpt_path)

        assert result["metadata"]["epoch"] == 5
        assert result["metadata"]["accuracy"] == 0.9

    def test_load_state_dict_missing_keys(self, tmp_path):
        """Test loading when checkpoint has missing keys."""
        model = SimpleModel()
        ckpt_path = tmp_path / "checkpoint.pt"

        # Save partial state dict
        partial_state = {"linear.weight": model.linear.weight}
        save_checkpoint(ckpt_path, partial_state)

        # Load (non-strict)
        new_model = SimpleModel()
        result = load_state_dict_safe(new_model, ckpt_path, strict_load=False)

        assert len(result["missing_keys"]) > 0


class TestGetCheckpointInfo:
    """Test getting checkpoint information."""

    def test_get_checkpoint_info_basic(self, tmp_path):
        """Test getting basic checkpoint info."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param": torch.randn(100, 100)}
        save_checkpoint(ckpt_path, state_dict)

        info = get_checkpoint_info(ckpt_path)

        assert info["exists"] is True
        assert info["size_mb"] > 0
        assert "hash" in info
        assert "keys" in info
        assert "state_dict" in info["keys"]

    def test_get_checkpoint_info_with_hash_file(self, tmp_path):
        """Test getting info when hash file exists."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param": torch.randn(10)}
        save_checkpoint(ckpt_path, state_dict, save_hash=True)

        info = get_checkpoint_info(ckpt_path)

        assert info["has_hash_file"] is True
        assert info["hash_valid"] is True

    def test_get_checkpoint_info_with_metadata(self, tmp_path):
        """Test getting info includes metadata."""
        ckpt_path = tmp_path / "checkpoint.pt"
        state_dict = {"param": torch.randn(10)}
        metadata = {"model": "test", "version": "1.0"}
        save_checkpoint(ckpt_path, state_dict, metadata=metadata)

        info = get_checkpoint_info(ckpt_path)

        assert "metadata" in info
        assert info["metadata"]["model"] == "test"
        assert info["metadata"]["version"] == "1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
