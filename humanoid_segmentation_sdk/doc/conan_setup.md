# Install the conan
pip install --upgrade conan
conan --version
conan profile detect
conan config set tools.system.package_manager:mode=install

#
conan cache clean --build

conan remove "zlib/1.3.1:*" -c
conan remove "zlib/*" -c

 9641  conan cache clean "*"
 9642  conan cache clean "*" --download
conan remove "*" -c
conan cache clean "*" --download
conan cache clean "*" --source
conan cache clean "*" --build


rm -rf ~/.conan2
