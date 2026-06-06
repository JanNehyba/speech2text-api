#!/usr/bin/env python3

from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="UTF-8") as f:
    long_description = f.read()

setup(
    name="speech2text-api",
    description="Speech2Text API",
    long_description=long_description,
    author="Michal Stefanik",
    author_email="stefanik.m@mail.muni.cz",
    packages=find_packages("src"),
    package_dir={"": "src"},
    use_scm_version={"write_to": ".version", "write_to_template": "{version}\n"},
    setup_requires=["setuptools_scm"],
    install_requires=[
        "click>=8.1",
        "email-validator>=2.0",
        "fastapi>=0.103,<0.116",
        "pydantic[dotenv]>=1.10.13,<2",
        "sentry-sdk>=1.40",
        "starlette>=0.27,<0.40",
        "statsd>=4.0",
        "uvicorn>=0.23",
        "faster-whisper>=1.1.0",
        "transformers>=4.45,<5",
        "python-multipart>=0.0.7",
        "librosa>=0.10",
        "numpy>=1.26,<2",
    ],
    extras_require={
        # Speaker diarization stack. Optional so `pip install .` for tests
        # of the transcription path stays light; the Docker build pulls
        # this in explicitly via requirements.txt.
        "diarize": [
            # pyannote.audio 4.0 ships the community-1 diarization pipeline
            # with VBx clustering (DER -5.9% on CALLHOME vs 3.1) and the
            # new DiarizeOutput dataclass exposing both regular and
            # exclusive_speaker_diarization annotations. See diarize.py.
            "pyannote.audio>=4.0,<5",
            "torch>=2.0,<3",
            "torchaudio>=2.0,<3",
        ],
    },
    entry_points={
        "console_scripts": ["speech2text-api=speech2text_api.__main__:main"]
    },
    package_data={"speech2text_api": ["py.typed"]},
    include_package_data=True,
    platforms=["platform-independent"],
)
