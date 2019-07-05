'''
setup.py for tatk
'''
import sys
import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

class LibTest(TestCommand):
	def run_tests(self):
		# import here, cause outside the eggs aren't loaded
		ret = os.system("pytest --cov=tatk tests/ --cov-report term-missing")
		sys.exit(ret >> 8)

setup(
	name='tatk',
	version='0.0.1',
	packages=find_packages(exclude=[]),
	license='Apache',
	description='Task-oriented Dialog System Toolkits',
	long_description=open('README.md', encoding='UTF-8').read(),
	long_description_content_type="text/markdown",
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'License :: OSI Approved :: Apache Software License',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
	],
	install_requires=[
		#'numpy>=1.15.0', # numpy应该可以不写吧，tf应该依赖numpy吧，让pip装tf时自己选个numpy装
		'nltk>=3.4',
		'tqdm>=4.30',
		'checksumdir>=1.1',
		'tensorflow==1.14',
		'scikit-learn>=0.20.3',
		'scipy>=1.2.1',
		#'allennlp>=0.8.2',
		'tensorboard>=1.14.0',
		#'tensorboardX==1.7',
		'requests'
	],
	extras_require={
		'develop':  [
			"python-coveralls",
			"pytest-dependency",
			"pytest-mock",
			"requests-mock",
			"pytest>=3.6.0",
			"pytest-cov==2.4.0",
			"checksumdir"
		]
	},
	cmdclass={'test': LibTest},
	entry_points={
		'console_scripts': [
			"tatk-report=tatk.scripts:report"
		]
	},
	include_package_data=True,
	url='https://github.com/thu-coai/tatk',
	author='thu-coai',
	author_email='thu-coai-developer@googlegroups.com',
	python_requires='>=3.5',
	zip_safe=False
)
