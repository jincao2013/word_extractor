from setuptools import setup, find_packages

setup(
    name='word_extractor',
    version='0.1',
    packages=['corpora', 'scripts', 'example'],
    # packages=find_packages(),
    scripts=["scripts/word_extractor"],
    include_package_data=True,
    install_requires=['nltk'],
    url='',
    license='',
    author='Jin Cao',
)