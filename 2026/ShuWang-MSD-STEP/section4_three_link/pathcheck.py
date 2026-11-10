"""Windows path-length check for the Numba cache.

nbody_common.py and parity_common.py put the Numba cache in <folder>/.numba_cache/. Numba names a cached function file like
.numba_cache/src_<40 hex digits>/jt_checked.quat_to_rotmat_numba-42.py39.1.nbc.tmp.<16 hex digits>, so the full path is
about 135 characters longer than the section folder. Without long-path support, Windows rejects paths of 260 characters
or more, and Numba then fails with FileNotFoundError while writing its cache. Keep the section folder at most 115
characters long (for example, clone the repository to C:\\src\\STEPv2-repro), or enable long paths in Windows.
"""
import os

LIMIT = 115


def long_paths_enabled():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\FileSystem')
        return winreg.QueryValueEx(k, 'LongPathsEnabled')[0] == 1
    except OSError:
        return False


def check(folder):
    if os.name != 'nt' or long_paths_enabled():
        return
    n = len(str(folder))
    if n > LIMIT:
        raise SystemExit(f'{folder}\nis {n} characters long; without Windows long-path support the Numba cache paths below it exceed '
                         f'260 characters and the run fails. Move the repository to a shorter path (section folder at most {LIMIT} '
                         'characters) or enable long paths (LongPathsEnabled = 1).')
