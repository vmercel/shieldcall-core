from setuptools import setup, find_packages

setup(
    name="shieldcall-core",
    version="0.3.0",
    description="Streaming dual-stream fraud + vocoder-artifact detection for telephone conditions",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "soundfile>=0.12",
    ],
    extras_require={
        "audio": ["librosa>=0.10"],
        "torch": ["torch>=2.0", "torchaudio>=2.0"],
        "dev": ["pytest>=7.0", "pytest-cov>=4.0", "matplotlib>=3.7"],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "shieldcall-demo=shieldcall.demo.stream_demo:main",
        ],
    },
)
