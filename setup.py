from setuptools import setup, find_packages

setup(
    name="heterofed-ids",
    version="1.0.0",
    author="Aniket Gundecha",
    author_email="author@mitaoe.ac.in",
    description="HeteroFed-IDS: Personalised Asynchronous Federated IDS for Heterogeneous IoT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/heterofed-ids",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "flwr>=1.8.0",
        "scipy>=1.11.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
