from setuptools import setup, find_packages

setup(
  name = 'dalle-pytorch-afiaka',
  packages = find_packages(),
  include_package_data = True,
  version = '0.0.1',
  license='MIT',
  description = 'highly opinionated refactor. go away.',
  author = 'afiaka87',
  author_email = 'samsepiol@gmail.com',
  url = 'https://github.com/afiaka87/dalle-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'text-to-image'
  ],
  install_requires=[
    'axial_positional_embedding',
    'DALL-E',
    'einops>=0.3',
    'ftfy',
    'g-mlp-pytorch',
    'pillow',
    'regex',
    'taming-transformers',
    'tokenizers',
    'torch>=1.7',
    'torchvision',
    'transformers',
    'tqdm',
    'wandb',
    'wheel',
    'Cython',
    'deepspeed',
    'youtokentome',
    'yapf',
    'WebDataset'
    'Ninja'
  ],
  classifiers=[
    'Development Status :: 4 - Alpha',
    'Intended Audience :: Me Currently',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)
