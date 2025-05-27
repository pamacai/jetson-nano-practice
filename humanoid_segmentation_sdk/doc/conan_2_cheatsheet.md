# Conan 2.x Cheatsheet

A quick reference for using [Conan](https://conan.io) 2.x — the C++ package manager.

---

## ✅ Installing Conan

```bash
pip install conan
```

---

## ✅ Creating a New Conan Project

```bash
conan new mylib/1.0 --layout cmake_lib
```

Creates:
- `conanfile.py`
- `src/`, `include/`
- `CMakeLists.txt`

---

## ✅ Installing Dependencies

```bash
conan install . --output-folder=build --build=missing
```

- `--build=missing`: Build from source if binaries not available.
- Requires `conanfile.py` or `conanfile.txt`.

---

## ✅ Building with CMake

```bash
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
```

---

## ✅ Consuming Packages

### `conanfile.txt` Example:
```ini
[requires]
fmt/10.1.1

[generators]
CMakeToolchain
CMakeDeps
```

---

### `conanfile.py` Example (2.x Style):
```python
from conan import ConanFile

class HelloConan(ConanFile):
    name = "hello"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    requires = "fmt/10.1.1"
    generators = "CMakeToolchain", "CMakeDeps"
```

---

## ✅ Creating a Package

```bash
conan create . --version=1.0
```

---

## ✅ Uploading to Remote

```bash
conan remote add myremote https://myrepo.com
conan upload hello/1.0@ --remote=myremote --all --confirm
```

---

## ✅ Listing Packages

```bash
conan search fmt
conan list fmt* -r=conancenter
```

---

## ✅ Exporting Local Recipes

```bash
conan export . hello/1.0@
```

---

## ✅ Clean Conan Cache

```bash
conan remove "*" --src --build --package --confirm
```

---

## 🔗 Useful Conan 2.x Docs
- https://docs.conan.io/2/

