from setuptools import find_packages, setup

package_name = "incremental_vo_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "requirements.txt"]),
    ],
    install_requires=["setuptools", "numpy", "opencv-python", "scipy"],
    zip_safe=True,
    maintainer="maintainer",
    maintainer_email="user@example.com",
    description="Skeleton incremental VO ROS 2 node",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "incremental_vo_node = incremental_vo_ros2.incremental_vo_node:main",
        ],
    },
)
