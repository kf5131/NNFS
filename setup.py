from setuptools import setup, find_packages

setup(
    name="nnfs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    author="Your Name",
    description="A neural network implementation from scratch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)
