import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastwedge",
    version="1.0.0",
    author="hari64",
    author_email="hari64boli64@gmail.com",
    license='MIT',
    url="https://github.com/hari64boli64/fastwedge",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
