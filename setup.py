from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='research_tools',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Container for functions and equations",
    license="GNUv3",
    author="Jacob Clenney",
    author_email='j2clenney@gmail.com',
    url='https://github.com/sivart213/research_tools',
    packages=['research_tools'],
    entry_points={
        'console_scripts': [
            'research_tools=research_tools.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='research_tools',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
