"""Unit tests for data pipeline."""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import ChangeDetectionDataset
from data.episode_sampler import EpisodeSampler, FewShotBatch
from data.transforms import get_train_transforms, get_val_transforms


class DummyDataset:
    """Dummy dataset for testing."""
    
    def __init__(self, num_samples=100, has_positive=True):
        self.num_samples = num_samples
        self.has_positive = has_positive
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img_t1 = torch.randn(3, 64, 64)
        img_t2 = torch.randn(3, 64, 64)
        
        # Make some samples have positive masks
        if self.has_positive and idx < self.num_samples // 2:
            mask = torch.zeros(64, 64)
            mask[10:30, 10:30] = 1.0
        else:
            mask = torch.zeros(64, 64)
        
        return {
            "img_t1": img_t1,
            "img_t2": img_t2,
            "mask": mask,
            "name": f"sample_{idx}",
        }


class TestEpisodeSampler:
    """Test episode sampler."""
    
    def test_sample_episode(self):
        """Test sampling an episode."""
        dataset = DummyDataset(100)
        sampler = EpisodeSampler(
            dataset,
            num_support=5,
            num_query=15,
            num_episodes=10,
        )
        
        episode = sampler.sample_episode()
        
        assert isinstance(episode, FewShotBatch)
        assert episode.support_t1.shape[0] >= 1
        assert episode.query_t1.shape[0] >= 1
    
    def test_iteration(self):
        """Test iteration over episodes."""
        dataset = DummyDataset(50)
        sampler = EpisodeSampler(dataset, num_episodes=5)
        
        episodes = list(sampler)
        assert len(episodes) == 5
    
    def test_move_to_device(self):
        """Test moving batch to device."""
        dataset = DummyDataset(50)
        sampler = EpisodeSampler(dataset, num_episodes=1)
        
        episode = sampler.sample_episode()
        
        # Move to CPU (always available)
        episode_cpu = episode.to(torch.device("cpu"))
        assert episode_cpu.support_t1.device.type == "cpu"


class TestTransforms:
    """Test data transforms."""
    
    def test_train_transform(self):
        """Test training transforms."""
        transform = get_train_transforms(image_size=(256, 256))
        
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 1
        
        result = transform(image=img1, image2=img2, mask=mask)
        
        assert "image" in result
        assert "image2" in result
        assert "mask" in result
        
        # Check output types
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (3, 256, 256)
    
    def test_val_transform(self):
        """Test validation transforms."""
        transform = get_val_transforms(image_size=(512, 512))
        
        img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        result = transform(image=img1, image2=img2, mask=mask)
        
        assert result["image"].shape == (3, 512, 512)
    
    def test_resize(self):
        """Test resizing in transforms."""
        transform = get_val_transforms(image_size=(256, 256))
        
        # Input is larger
        img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        result = transform(image=img1, image2=img2, mask=mask)
        
        assert result["image"].shape == (3, 256, 256)
        assert result["mask"].shape == (256, 256)


class TestChangeDetectionDataset:
    """Test base dataset class."""
    
    def test_with_temp_dir(self):
        """Test dataset with temporary directory."""
        import cv2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure
            os.makedirs(os.path.join(tmpdir, "A"))
            os.makedirs(os.path.join(tmpdir, "B"))
            os.makedirs(os.path.join(tmpdir, "label"))
            
            # Create dummy files
            for i in range(3):
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                mask = np.zeros((64, 64), dtype=np.uint8)
                
                cv2.imwrite(os.path.join(tmpdir, "A", f"{i:03d}.png"), img)
                cv2.imwrite(os.path.join(tmpdir, "B", f"{i:03d}.png"), img)
                cv2.imwrite(os.path.join(tmpdir, "label", f"{i:03d}.png"), mask)
            
            dataset = ChangeDetectionDataset(tmpdir, split="all")
            
            assert len(dataset) == 3
            
            sample = dataset[0]
            assert "img_t1" in sample
            assert "img_t2" in sample
            assert "mask" in sample


class TestFewShotBatch:
    """Test FewShotBatch dataclass."""
    
    def test_creation(self):
        """Test batch creation."""
        batch = FewShotBatch(
            support_t1=torch.randn(5, 3, 64, 64),
            support_t2=torch.randn(5, 3, 64, 64),
            support_masks=torch.zeros(5, 64, 64),
            query_t1=torch.randn(10, 3, 64, 64),
            query_t2=torch.randn(10, 3, 64, 64),
            query_masks=torch.zeros(10, 64, 64),
        )
        
        assert batch.support_t1.shape[0] == 5
        assert batch.query_t1.shape[0] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
