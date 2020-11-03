import sys
from project import python_version


def main():
    major, minor, micro = [int(s) for s in python_version.split('.')]
    sys_major = sys.version_info.major
    sys_minor = sys.version_info.minor
    if sys_major < major:
      raise TypeError(f"You are running Python {sys_major} - you need to be using {major} or above!")
    if sys_minor < minor:
        raise TypeError(f"You are running Python {sys_major}.{sys_minor} - you need to be using {major}.{minor} or above!")
    print(f"You are using Python {sys_major}.{sys_minor} - development environment passes all tests!")


if __name__ == '__main__':
    main()
