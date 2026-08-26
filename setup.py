from setuptools import setup, find_packages

setup(
    name="shieldcall-core",
    version="0.6.0",
    description="Streaming dual-stream fraud + vocoder-artifact detection for telephone conditions",
    packages=find_packages(),
    include_package_data=True,
    package_data={"shieldcall.serve": ["static/*"]},
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
        "serve": ["fastapi>=0.110", "uvicorn>=0.27", "httpx>=0.27", "python-multipart>=0.0.9"],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "shieldcall-demo=shieldcall.demo.stream_demo:main",
        ],
    },
)
