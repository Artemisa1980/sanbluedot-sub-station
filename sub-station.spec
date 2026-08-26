# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['substation_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('translation_worker.py', '.'),
        ('assets/sanbluedot-original.png', 'assets'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sub-station',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='sub-station',
)
app = BUNDLE(
    coll,
    name='sub-station.app',
    icon='assets/sanbluedot-original.png',
    bundle_identifier='com.sanbluedot.substation',
    version='2.3.1',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'NSHumanReadableCopyright': '© 2026 Sandy E. Quintero',
    },
)
