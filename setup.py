"""Setup script for causal-orchestration-mhealth package."""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="causal-orchestration-mhealth",
    version="1.0.0",
    description="Multi-agent orchestration for automated clinical care plan generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sanjaybasu/causal-orchestration-mhealth",
    author="Sanjay Basu, Aaron Baum",
    author_email="sanjay.basu@waymarkcare.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="healthcare, AI, multi-agent systems, LLM, clinical decision support",
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.10, <4",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
    ],
    project_urls={
        "Bug Reports": "https://github.com/sanjaybasu/causal-orchestration-mhealth/issues",
        "Source": "https://github.com/sanjaybasu/causal-orchestration-mhealth",
    },
)
