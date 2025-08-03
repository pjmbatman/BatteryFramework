"""
Setup script for Battery Foundation Model Framework
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Battery Foundation Model Framework - A professional framework for battery data analysis and modeling"

# Read version from package
VERSION = "1.0.0"

# Define requirements
INSTALL_REQUIRES = [
    "torch>=1.12.0",
    "numpy>=1.21.0",
    "pyyaml>=6.0",
    "tqdm>=4.64.0",
    "scikit-learn>=1.1.0",
    "pandas>=1.4.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "pre-commit>=2.17.0",
    ],
    "docs": [
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinx-autodoc-typehints>=1.17.0",
    ],
    "notebook": [
        "jupyter>=1.0.0",
        "ipywidgets>=7.7.0",
        "plotly>=5.8.0",
    ],
}

# All extras
EXTRAS_REQUIRE["all"] = sum(EXTRAS_REQUIRE.values(), [])

setup(
    name="battery-foundation",
    version=VERSION,
    author="Battery Foundation Team",
    author_email="battery.foundation@example.com",
    description="A professional framework for battery data analysis and modeling with LiPM integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/battery-foundation/battery-foundation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "battery-foundation=battery_foundation.cli:main",
            "bf=battery_foundation.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "battery_foundation": [
            "configs/*.yaml",
            "data/sample_data/*",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/battery-foundation/battery-foundation/issues",
        "Source": "https://github.com/battery-foundation/battery-foundation",
        "Documentation": "https://battery-foundation.readthedocs.io/",
    },
)