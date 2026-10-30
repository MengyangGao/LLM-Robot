from glob import glob
from setuptools import setup

setup(
    name='rosgpt', version='0.2.0', packages=['rosgpt'],
    data_files=[('share/ament_index/resource_index/packages', ['resource/rosgpt']),
                ('share/rosgpt', ['package.xml']),
                ('share/rosgpt/launch', glob('launch/*.launch.py'))],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='LLM-Robot contributors', maintainer_email='maintainer@example.com',
    description='Local vision-language planning for ROS 2 turtlesim',
    entry_points={'console_scripts': [
        'rosgpt = rosgpt.rosgpt:main',
        'rosgptparser_turtlesim = rosgpt.rosgptparser_turtlesim:main',
    ]},
)
