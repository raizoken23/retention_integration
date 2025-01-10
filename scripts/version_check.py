#!/usr/bin/env python3

import sys
import json
from urllib.request import urlopen

def check_version(local_version, package_name, test_pypi=False):
    registry = "TestPyPI" if test_pypi else "PyPI"
    base_url = "https://test.pypi.org" if test_pypi else "https://pypi.org"
    url = f"{base_url}/pypi/{package_name}/json"

    try:
        data = json.load(urlopen(url))
        latest = data['info']['version']
        print(f"{registry} version: {latest}")
        
        if local_version <= latest:
            print(f"\033[91mERROR: Local version {local_version} is not greater than {registry} version {latest}\033[0m")
            sys.exit(1)
            
    except Exception as e:
        if '404' in str(e):
            print(f'Package not found on {registry} (first release?)')
        else:
            print(f'Error checking {registry} version: {e}')
            sys.exit(1)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: version_check.py <version> <package_name> [--test]")
        sys.exit(1)
        
    version = args[0]
    package_name = args[1]
    test_pypi = "--test" in args
    
    check_version(version, package_name, test_pypi) 