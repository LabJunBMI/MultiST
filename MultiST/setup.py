from setuptools import setup

# with open("README.rst", "r", encoding="utf-8") as f:
#     __long_description__ = f.read()

if __name__ == "__main__":
    setup(
        name = "MultiST",
        version = "1.0.0",
        description = "MultiST cross-attention based multimodal model for spatial transcriptomics",
        url = "https://github.com/LabJunBMI",
        author = "Wei Wang",
        author_email = "wang3wa@mail.uc.edu",
        license = "MIT",
        packages = ["MultiST"],
        install_requires = ["requests"],
        zip_safe = False,
        include_package_data = True,
        long_description = """ Long Description """,
        long_description_content_type="text/markdown",
    )
