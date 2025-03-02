import os
from setuptools import setup, find_packages

with open("README.md",'r') as f:
    x = f.read()

setup(
    name = "tinyrag",
    version = "0.0.2",
    author = "divine-architect",
    author_email = "cybers.mail0@proton.me",
    description = ("Lightweight RAG implementation for all your LLM endpoints"),
    license = "MIT",
    keywords = "llm rag ai llms ml openai gpt claude ollama AI ML LLM",
    url = "https://github.com/divine-architect/tinyrag",
    packages=find_packages(),
    long_description=x,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires = [
        'numpy',
        'scikit-learn',
        'ollama'
    ]

)