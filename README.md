bdilab_detect_xdu

集成完新算法，打包、上传至Pypi。
1. 注册Pypi账号
2. 与setup.py处于同一目录
3. python setup.py sdist bdist_wheel (在dist目录下生成一个tar.gz的源码包和一个.whl的Wheel包)
4. python install twine (可选)
5. twine upload dist/* (此步骤要求输入Pypi账号、密码)