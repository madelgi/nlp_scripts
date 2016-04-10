try:
    from setuptools import setup
except ImportError:
    from docutils.core import setup

config = {
    'description': 'A collection of nlp algorithms.'
    'author': 'Max Del Giudice',
    'url': 'TODO',
    'author_email': 'maxdelgiudice@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['nlp_scripts'],
    'scripts': [],
    'name': 'nlp_scripts'
}

setup(**config)
