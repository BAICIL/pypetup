from setuptools import setup, find_packages
# Full setup configuration
setup(
    name='pypetup',  
    version='0.1',  
    license='MIT License',
    description='A tool to process PET images and perfrom RSF PVC correction.',  
    author='Dhruman Goradia, PhD', 
    author_email='Dhruman.Goradia2@bannerhealth.com',  
    url='https://github.com/BAICIL/pypetup',
    
    # Automatically find and include all packages in the project
    packages=find_packages(),  
    include_package_data=True,
    package_data={
        'pypetup':['*.json'],
    },

    # List dependencies (install_requires can also directly list dependencies if requirements.txt is not used)
    install_requires=[
        'fslpy',
        'matplotlib',
        'numpy',
        'nibabel',
        'pandas',
        'scipy'
    ],

    # Entry point configuration
    entry_points={
        'console_scripts': [
            'run_pup=pypetup.petproc:main',  # Links the CLI script 'convert-to-nifti' to the 'main' function in cli.py
        ],
    },

    # Additional metadata
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOSX',
        'Operating System :: Linux'
    ],
    
    python_requires='>=3.9',  # Specifies that Python 3.9 or newer is required
)