#!/bin/zsh

# SAM
git clone https://github.com/facebookresearch/segment-anything

cd segment-anything
mkdir -p ckpt
cd ckpt

if [ ! -f sam_vit_b_01ec64.pth ]; then
    curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

echo "Done!"
echo
