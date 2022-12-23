import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="segmentation-mask-overlay",
    version="0.3.3",
    description="Plotting the segmentation masks has never been so exciting!",
    long_description=README,
    long_description_content_type="text/markdown",
    author="armavox",
    author_email="armavox@gmail.com",
    license="MIT",
    url="https://github.com/lobantseff/segmentation-mask-overlay",
    packages=find_packages(include=["segmentation_mask_overlay"]),
    include_package_data=True,
    install_requires=[
        "matplotlib>=3.4.2",
        "numpy>=1.20",
        "opencv-python-headless>=4.5.3",
        "Pillow>=7.2.0"

    ],
    setup_requires=["flake8"],
)
