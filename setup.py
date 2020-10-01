from distutils.core import setup

setup(
    name='Tacotron2',
    version='0.0',
    install_requires=[
        'torch>=1.4.0',
        'librosa >= 0.7.0',
        'numpy',
        'pandas'
    ]
)
