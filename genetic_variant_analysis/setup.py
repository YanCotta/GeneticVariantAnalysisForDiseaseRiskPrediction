from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements-dev.txt') as f:
    dev_requirements = f.read().splitlines()

setup(
    name="genetic-variant-analysis",
    version="2.2.0",
    packages=find_packages(include=['genetic_variant_analysis', 'genetic_variant_analysis.*']),
    package_data={
        'genetic_variant_analysis': ['config/*.yaml'],
    },
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
        'gpu': ['torch>=1.8.0+cu111']
    },
    author="Yan Cotta",
    author_email="yanpcotta@gmail.com",
    description="Production-ready genetic variant analysis pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YanCotta/GeneticVariantAnalysisForDiseaseRiskPrediction",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ]
)
