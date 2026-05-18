from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lantern",
    version="0.1.0",
    author="LantErn Team",
    description="Interleaved Reasoning between text (verbalized form) and visual representations (non-verbalize forms)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GuilhermeViveiros/LantErn",
    packages=find_packages(),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.30.0",
        "deepspeed>=0.9.0",
        "termcolor",
        "qwen-vl-utils",
        "tqdm",
        "Pillow",
        "dataclasses; python_version<'3.7'",
    ],
    extras_require={
        "wandb": ["wandb"],
    },
    entry_points={
        "console_scripts": [
            "lantern-train=src.train.train:main",
        ],
    },
)
