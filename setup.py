from pathlib import Path

from setuptools import setup, find_packages


def read(relative: str) -> str:
    base: Path = Path(__file__).parent
    return base.joinpath(relative).read_text()


setup(
    name='detector',
    version='0.1.0',
    author='Tuna Alikasifoglu',
    author_email='tunakasif@gmail.com',
    packages=find_packages(),
    install_requires=["opencv-contrib-python", "imutils"],
)
