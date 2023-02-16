#!/bin/bash
rm -rf dist src/segmentation_mask_overlay.egg-info
python -m build
twine check dist/*
# twine upload dist/*