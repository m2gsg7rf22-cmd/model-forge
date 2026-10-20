"""Archive and round-trip-test the actual native app before hashing release assets."""

from pathlib import Path
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

root = Path(__file__).resolve().parents[1]
dist = root / "dist"
version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]["version"]
name = f"ModelForge-{version}-{platform.system()}-{platform.machine()}"
source = "ModelForge.app" if sys.platform == "darwin" else "ModelForge"
archive = dist / (name + ".zip")
if sys.platform == "darwin":
    # Framework symlinks must survive packaging; generic ZIP writers lose directory symlinks.
    subprocess.run(["xattr", "-cr", str(dist / source)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", str(dist / source)], check=True)
    if archive.exists():
        archive.unlink()
    subprocess.run(
        [
            "ditto",
            "-c",
            "-k",
            "--keepParent",
            "--norsrc",
            "--noextattr",
            str(dist / source),
            str(archive),
        ],
        check=True,
    )
else:
    shutil.make_archive(str(dist / name), "zip", root_dir=dist, base_dir=source)

with tempfile.TemporaryDirectory(prefix="modelforge-unpack-") as temporary:
    destination = Path(temporary)
    if sys.platform == "darwin":
        subprocess.run(
            ["ditto", "-x", "-k", str(archive), str(destination)], check=True
        )
        executable = destination / source / "Contents" / "MacOS" / "ModelForge"
        subprocess.run(
            ["codesign", "--verify", "--deep", "--strict", str(destination / source)],
            check=True,
        )
    else:
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
            if sys.platform != "win32":
                for item in zipped.infolist():
                    (destination / item.filename).chmod(
                        (item.external_attr >> 16) & 0o777
                    )
        executable = (
            destination
            / source
            / ("ModelForge.exe" if sys.platform == "win32" else "ModelForge")
        )
    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MODELFORGE_HOME=str(destination / "research-data"),
    )
    subprocess.run(
        [str(executable), "--smoke-test"], env=environment, check=True, timeout=60
    )

assets = sorted(
    p
    for p in dist.iterdir()
    if p.is_file() and p.name.endswith((".zip", ".whl", ".tar.gz", ".dmg"))
)
(dist / "SHA256SUMS.txt").write_text(
    "".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
        for p in assets
    )
)
print("Archive extracted and frozen app launch passed:", archive.name)
