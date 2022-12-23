#!/bin/bash
rm -rf build dist segmentation_mask_overlay.egg-info/
python3 setup.py sdist bdist_wheel
twine check dist/*