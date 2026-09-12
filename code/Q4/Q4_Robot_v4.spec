# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
q4_dir = Path(SPECPATH)
project_root = q4_dir.parent.parent
a = Analysis([str(q4_dir / "run_q4_live_v4.py")], pathex=[str(q4_dir), str(project_root / "code" / "Q3"), str(project_root / "code" / "Q2"), str(project_root / "code" / "Q1")], binaries=[], datas=[], hiddenimports=[], hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=["matplotlib", "pandas", "pytest", "torch", "tensorflow", "sympy", "jinja2", "fsspec", "lxml", "scipy"], noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="Q4_Robot_v4", debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True, disable_windowed_traceback=False, argv_emulation=False, target_arch=None, codesign_identity=None, entitlements_file=None)
