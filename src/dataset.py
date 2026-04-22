import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
import random


class RoofSolarDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, img_size=512, train_ratio=0.8):
        """
        Dataset for roof and solar panel segmentation

        Args:
            data_dir: Directory containing images and masks
            split: 'train' or 'val'
            transform: Optional augmentation transforms
            img_size: Target image size (will resize to this)
            train_ratio: Ratio of training data
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        # Get all images
        self.image_paths = sorted(list(self.data_dir.glob('*.jpg')))

        # Create mapping for mask files
        self.roof_masks = {}  # _2.npz files
        self.solar_masks = {}  # _3.npz files

        # Build mask dictionaries
        for npz_file in self.data_dir.glob('*.npz'):
            if str(npz_file).endswith('_2.npz'):
                # Extract base name for roof masks
                base_name = self._extract_base_name(npz_file.name, '_2.npz')
                self.roof_masks[base_name] = npz_file
            elif str(npz_file).endswith('_3.npz'):
                # Extract base name for solar masks
                base_name = self._extract_base_name(npz_file.name, '_3.npz')
                self.solar_masks[base_name] = npz_file

        # Filter images that have at least one mask
        self.valid_samples = []
        for img_path in self.image_paths:
            base_name = self._get_image_base_name(img_path.name)
            has_roof = base_name in self.roof_masks
            has_solar = base_name in self.solar_masks

            if has_roof or has_solar:
                self.valid_samples.append({
                    'image': img_path,
                    'roof_mask': self.roof_masks.get(base_name),
                    'solar_mask': self.solar_masks.get(base_name)
                })

        # Split into train/val
        random.seed(42)
        random.shuffle(self.valid_samples)
        n_train = int(len(self.valid_samples) * train_ratio)

        if split == 'train':
            self.samples = self.valid_samples[:n_train]
        else:
            self.samples = self.valid_samples[n_train:]

        print(f"{split} dataset: {len(self.samples)} samples")

        # Basic transforms
        self.resize = T.Resize((img_size, img_size))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    def _extract_base_name(self, filename, suffix):
        """Extract base name from mask filename"""
        # Remove suffix and convert format
        name = filename.replace(suffix, '')
        # Convert from mask format to image format
        # area-XXXX-date-YYYY-MM-DD-labels-all-instance-mask_dk -> area_XXXX_YYYY-MM-DD
        parts = name.split('-')
        if len(parts) >= 5:
            area = parts[0]
            area_id = parts[1]
            date = '-'.join(parts[3:6])  # YYYY-MM-DD
            return f"{area}_{area_id}_{date}"
        return name

    def _get_image_base_name(self, img_filename):
        """Get base name from image filename"""
        # area_XXXX_YYYY-MM-DD.jpg -> area_XXXX_YYYY-MM-DD
        return img_filename.replace('.jpg', '')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image']).convert('RGB')
        image = self.resize(image)

        # Initialize masks and valid masks
        h, w = self.img_size, self.img_size
        roof_mask = np.zeros((h, w), dtype=np.float32)
        solar_mask = np.zeros((h, w), dtype=np.float32)
        roof_valid = np.zeros((h, w), dtype=np.float32)
        solar_valid = np.zeros((h, w), dtype=np.float32)

        # Load roof mask if exists
        if sample['roof_mask'] is not None:
            roof_data = np.load(sample['roof_mask'])['dkmask']
            # Resize mask to match image
            roof_data = self._resize_mask(roof_data, (h, w))
            # Convert to binary labels
            roof_mask = (roof_data >= 1).astype(np.float32)
            # Valid mask (not -1)
            roof_valid = (roof_data != -1).astype(np.float32)

        # Load solar mask if exists
        if sample['solar_mask'] is not None:
            solar_data = np.load(sample['solar_mask'])['dkmask']
            # Resize mask to match image
            solar_data = self._resize_mask(solar_data, (h, w))
            # Convert to binary labels
            solar_mask = (solar_data >= 1).astype(np.float32)
            # Valid mask (not -1)
            solar_valid = (solar_data != -1).astype(np.float32)

        # Color jitter on PIL image (before normalisation)
        if self.transform and self.split == 'train':
            image = self.transform.color_jitter(image)

        # Convert image to tensor and normalize
        image = self.to_tensor(image)
        image = self.normalize(image)

        # Stack masks: [2, H, W]
        masks = torch.stack([
            torch.from_numpy(roof_mask),
            torch.from_numpy(solar_mask)
        ])

        valid_masks = torch.stack([
            torch.from_numpy(roof_valid),
            torch.from_numpy(solar_valid)
        ])

        # Spatial augmentations on tensors (image + masks together)
        if self.transform and self.split == 'train':
            image, masks, valid_masks = self.transform.spatial_flip(image, masks, valid_masks)

        return {
            'image': image,
            'masks': masks,
            'valid_masks': valid_masks,
            'image_path': str(sample['image'])
        }

    def _resize_mask(self, mask, target_size):
        """Resize mask using nearest neighbor interpolation"""
        h, w = target_size
        mask_pil = Image.fromarray(mask.astype(np.float32))
        mask_resized = mask_pil.resize((w, h), Image.NEAREST)
        return np.array(mask_resized)


class JointAugmentation:
    """Augmentation that applies the same spatial transforms to image, masks, and valid_masks."""

    def __init__(self):
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def spatial_flip(self, image, masks, valid_masks):
        # Random horizontal flip (image + masks together)
        if random.random() > 0.5:
            image = torch.flip(image, [-1])
            masks = torch.flip(masks, [-1])
            valid_masks = torch.flip(valid_masks, [-1])

        # Random vertical flip (image + masks together)
        if random.random() > 0.5:
            image = torch.flip(image, [-2])
            masks = torch.flip(masks, [-2])
            valid_masks = torch.flip(valid_masks, [-2])

        return image, masks, valid_masks


def get_augmentation():
    """Get augmentation transforms for training"""
    return JointAugmentation()


if __name__ == "__main__":
    # Test the dataset
    dataset = RoofSolarDataset(
        data_dir="MLtestdata",
        split='train',
        transform=get_augmentation(),
        img_size=896
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Masks shape: {sample['masks'].shape}")
    print(f"Valid masks shape: {sample['valid_masks'].shape}")
    print(f"Roof pixels: {sample['masks'][0].sum().item()}")
    print(f"Solar pixels: {sample['masks'][1].sum().item()}")
    print(f"Valid roof pixels: {sample['valid_masks'][0].sum().item()}")
    print(f"Valid solar pixels: {sample['valid_masks'][1].sum().item()}")