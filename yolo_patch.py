import torch
import torch.nn as nn
from einops import rearrange


def create_random_mask(prob, size):
    """
    prob: masking될 비율
    size: masking tensor의 크기
    """
    mask = torch.rand(size)

    # prob만큼의 비율로 1로 설정
    mask = mask > prob

    return mask


class YOLOPATCH(nn.Module):
    def __init__(self, patch_masking):
        self.patch_masking = patch_masking
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, images):
        """
        :images: (batch_size, 3, 800, 1024)
        )"""
        patches = rearrange(images, 'b c (h p1) (w p2) -> (b p1 p2) c h w', p1=32, p2=32)
        if self.patch_masking:
            batch_size = patches.shape[0]
            mask = create_random_mask(0.3, batch_size)
            patches[mask, :, :, :] = 0
        patches = self.model(patches)
        output = rearrange(patches, '(b p1 p2) c h w -> b c (h p1) (w p2)', p1=32, p2=32)
        return output


if __name__ == "__main__":
    model = YOLOPATCH(True).cuda()
    images = torch.randn(10, 3, 800, 1024).cuda()
    output = model(images)
