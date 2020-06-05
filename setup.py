from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name='facenet_models',
        version="1.0",
        author="David Mascharka",
        description="Manages facenet_pytorch's face detection/recognition models",
        license="MIT",
        platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
        packages=find_packages(where="src/"),
        package_dir={"": "src"},
    )
