from setuptools import setup, find_packages

setup(
    name="deepgis",
    version="0.1.0",
    description="A deep learning library for geospatial image classification",
    author="Ziwei Xiang, Jianhang Zhang, Runbao Zhang, Zhongyu Xiu, Xupeng Wei, Yujiao Gao",
    author_email="ziweixiang@u.nus.edu",
    url="https://github.com/Knight0132/deepgis",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "rasterio>=1.2.0",
        "scikit-learn>=0.24.0",
        "pillow>=8.0.0",
    ],
    python_requires=">=3.7",
)