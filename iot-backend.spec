# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['app.main']
hiddenimports += collect_submodules('app')
hiddenimports += collect_submodules('assistant')
hiddenimports += collect_submodules('nlp')


a = Analysis(
    ['packaging/backend/sidecar_entry.py'],
    pathex=[],
    binaries=[],
    datas=[('app', 'app'), ('assistant', 'assistant'), ('nlp', 'nlp'), ('artifacts/models/fd001_lstm_baseline/best_model.pt', 'artifacts/models/fd001_lstm_baseline'), ('artifacts/data/fd001_lstm_baseline/scaler.joblib', 'artifacts/data/fd001_lstm_baseline'), ('artifacts/embeddings/fd001_lstm_baseline/sensor_embeddings.npy', 'artifacts/embeddings/fd001_lstm_baseline')],
    hiddenimports=hiddenimports,
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
    a.binaries,
    a.datas,
    [],
    name='iot-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
