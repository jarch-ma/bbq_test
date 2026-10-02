"""Package the LIT workflow; install the bundled SAM 3 source alongside it."""

from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="lit-sam3",
    version="1.0.0",
    description="Live Interactive Training for SAM 3 video segmentation",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["lit_sam3", "lit_sam3.*"]),
    python_requires=">=3.10",
    install_requires=[
        "sam3",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "numpy>=1.26,<2",
        "pillow>=9.4.0",
        "tqdm>=4.66.1",
        "opencv-python<4.12",
        "pycocotools>=2.0.8",
        "pandas>=2.2.2",
        "einops",
        "scipy",
        "eva-decord>=0.6.1",
        "psutil",
        "packaging",
        "setuptools<81",  # The bundled SAM 3 builder imports pkg_resources.
    ],
    extras_require={
        "notebooks": ["matplotlib", "jupyter"],
        "dev": ["pytest"],
    },
)
