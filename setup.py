from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hathor",
    version="0.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyautogen",
        "openai",
    ],
    python_requires=">=3.9",  
    author="Autumn Pearce, Yevhen Kylivnyk"
    project_urls={
        "Bug Reports": "https://github.com/AutumnPearce/hathor/issues",
        "Source": "https://github.com/AutumnPearce/hathor",
    },
    description="Multi-agent system for generating galaxy formation hypotheses and creating relevant plots from RAMSES simulation data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy ",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)