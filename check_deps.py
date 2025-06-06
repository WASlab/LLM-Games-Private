# check_missing_deps.py
import importlib

missing = []
with open('requirements.txt') as f:
    for line in f:
        if not line.strip() or line.startswith('#'):
            continue
        pkg = line.split('==')[0].replace('-', '_').replace('[', '').replace(']', '')
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

print("Missing packages:", missing)
