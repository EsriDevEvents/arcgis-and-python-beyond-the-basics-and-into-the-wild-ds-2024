# -*- mode: python ; coding: utf-8 -*-
import flask_bootstrap
import arcgis
import pathlib
block_cipher = None


a = Analysis(
    ['desktop.py'],
    pathex=[],
    binaries=[],
    datas=[('templates', 'templates'), ('static', 'static'), (f"{pathlib.Path(flask_bootstrap.__file__).parent}", "flask_bootstrap"), (f"{pathlib.Path(arcgis.__file__).parent}", "arcgis")],
    hiddenimports=['flask_bootstrap'],
    hookspath=['.\\hook-flask-bootstrap.py'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set to True to see console output for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
