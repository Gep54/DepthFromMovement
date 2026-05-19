from setuptools import find_packages, setup

package_name = "map_3d_ros2"

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
    description="Sparse 3D map ROS 2 node (map_3d)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "map_3d = map_3d_ros2.map_3d_node:main",
        ],
    },
)
