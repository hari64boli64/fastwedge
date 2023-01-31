import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="myawsname",
    version="0.0.1",
    author="cm-taro",
    author_email="cmtaro@example.com",
    license='MIT',
    description="You can receive AWS Service Name.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cm-ichida/myawsname",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
