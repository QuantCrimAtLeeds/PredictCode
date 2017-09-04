from setuptools import setup

def find_version():
    import os
    with open(os.path.join("open_cp", "__init__.py")) as file:
        for line in file:
            if line.startswith("__version__"):
                start = line.index('"')
                end = line[start+1:].index('"')
                return line[start+1:][:end]
            
long_description = ""
            
setup(
    name = 'opencp',
    packages = ['open_cp'],
    version = find_version(),
    install_requires = [], # TODO
    python_requires = '>=3.5',
    description = 'Standardised implementations of crime prediction techniques in the literature',
    long_description = long_description,
    author = 'QuantCrimAtLeeds',
    author_email = 'matthew.daws@gogglemail.com',
    url = 'https://github.com/QuantCrimAtLeeds/PredictCode',
    license = 'Artistic',
    keywords = [],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Artistic License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: GIS"
    ]
)