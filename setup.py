#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Flash Attention V100

This package provides an optimized implementation of Flash Attention
for V100 GPUs using Triton kernels.
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Version
__version__ = "1.0.0"

# Read the README file
def read_readme():
    """Read README.md file"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    req_path = Path(__file__).parent / filename
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Check CUDA availability
def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def check_compute_capability():
    """Check GPU compute capability"""
    try:
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return capability >= (7, 0)  # V100 is 7.0
        return False
    except ImportError:
        return False

# Custom build command to check requirements
class CustomBuildExt(build_ext):
    def run(self):
        # Skip dependency checks during build to avoid space issues
        super().run()

# Long description
long_description = read_readme()
if not long_description:
    long_description = """
Flash Attention V100

An optimized implementation of Flash Attention for V100 GPUs using Triton kernels.

Features:
- Memory-efficient attention computation
- Support for causal and non-causal attention
- Gradient computation for training
- Automatic kernel tuning for optimal performance
- Multi-GPU support
- Enhanced numerical stability
"""

# Requirements
install_requires = [
    "torch>=2.0.0",
    "triton>=2.1.0",
    "numpy>=1.20.0",
    "psutil>=5.8.0",
]

# Optional requirements
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-xdist>=2.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.900",
    ],
    "test": [
        "pytest>=6.0.0",
        "pytest-xdist>=2.0.0",
        "pytest-benchmark>=3.0.0",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=0.15.0",
    ],
}

# All extra requirements
extras_require["all"] = sum(extras_require.values(), [])

# Package data
package_data = {
    "flash_attention_v100": [
        "*.py",
        "kernels/*.py",
        "ops/*.py",
        "tests/*.py",
        "examples/*.py",
    ]
}

# Entry points
entry_points = {
    "console_scripts": [
        "flash-attn-benchmark=flash_attention_v100.examples.basic_usage:main",
    ],
}

# Classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Setup configuration
setup(
    name="flash-attention-v100",
    version=__version__,
    description="Optimized Flash Attention implementation for V100 GPUs using Triton",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MINI_kio",
    author_email="kiolaaoz@naver.com",
    url="https://github.com/mini-kio/flash-attn-v100",
    license="Apache 2.0",
    
    # Packages
    packages=["flash_attention_v100", "flash_attention_v100.kernels", "flash_attention_v100.ops"],
    package_dir={
        "flash_attention_v100": ".",
        "flash_attention_v100.kernels": "kernels",
        "flash_attention_v100.ops": "ops"
    },
    py_modules=[],
    package_data=package_data,
    include_package_data=True,
    
    # Requirements
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points
    entry_points=entry_points,
    
    # Build configuration
    cmdclass={"build_ext": CustomBuildExt},
    
    # Metadata
    classifiers=classifiers,
    keywords="attention, transformer, pytorch, triton, v100, gpu",
    project_urls={
        "Bug Reports": "https://github.com/mini-kio/flash-attn-v100/issues",
        "Source": "https://github.com/mini-kio/flash-attn-v100",
        "Documentation": "https://flash-attn-v100.readthedocs.io/",
    },
    
    # Additional options
    zip_safe=False,
    platforms=["Linux"],
)
