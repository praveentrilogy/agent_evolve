"""
Setup script for Agent Evolve package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agent-evolve",
    version="1.0.0",
    author="Agent Evolution Team",
    description="A comprehensive toolkit for evolving and tracking AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add any required dependencies here
        # "langchain-openai>=0.1.0",
        # "openevolve>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "agent-evolve=agent_evolve.__main__:main",
        ],
    },
)