from setuptools import setup, find_packages

setup(
    name="shieldcall-core",
    version="0.3.0",
    description="Streaming dual-stream fraud + deepfake detection for telephone conditions",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
    ],
    extras_require={
        "audio": ["librosa>=0.10", "soundfile>=0.12"],
        "torch": ["torch>=2.0", "torchaudio>=2.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.6.0",
            "mypy>=1.11.0",
            "pip-audit>=2.7.0",
        ],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "shieldcall-demo=shieldcall.demo.stream_demo:main",
        ],
    },
)
