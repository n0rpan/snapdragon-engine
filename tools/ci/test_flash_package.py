#!/usr/bin/env python3
"""Offline packaging fixtures: real scripts/tar, fake binary and metadata tools."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PINS = {'BigMoeOnEdge': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
        'llama.cpp': '94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d'}


class FlashPackage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'repo'
        for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES', 'vendor/ggml-subset/LICENSE',
                     'vendor/ggml-flash/LICENSE', 'third_party/stable-diffusion.cpp-LICENSE',
                     'tools/ci/package.sh', 'tools/ci/crate_licenses.py', 'tools/ci/flash_licenses.py'):
            dest = self.root / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        self.sources = []
        self.expected = {}
        for name, pin in PINS.items():
            raw = (ROOT / 'third_party' / (name + '-LICENSE')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin)
            source = Path(self.tmp.name) / name; source.mkdir()
            (source / 'LICENSE').write_bytes(raw); self.sources.append(source); self.expected[name] = raw
        self.prod = Path(self.tmp.name) / 'prod'
        self.collect()
        (self.prod / 'MANIFEST').write_text('fixture: no native libraries built or linked\n')
        self.target = Path(self.tmp.name) / 'target/release'; self.target.mkdir(parents=True)
        (self.target / 'oxe-server').write_text('package fixture, not a runnable binary\n')
        self.dist = Path(self.tmp.name) / 'dist'
        bindir = Path(self.tmp.name) / 'stub-tools'; bindir.mkdir()
        scripts = {
            'cargo': '#!/usr/bin/env python3\nimport json,sys\nif sys.argv[1]=="metadata": print(json.dumps({"workspace_members":[],"packages":[]}))\n',
            'rustc': '#!/bin/sh\necho "rustc fixture"\n',
            'cc': '#!/bin/sh\necho "cc fixture"\n',
            'objdump': '#!/bin/sh\necho "fixture GLIBC_2.39"\n',
        }
        for name, text in scripts.items():
            p = bindir / name; p.write_text(text); p.chmod(0o700)
        self.env = {**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
                    'OXF_GGML_PROD': str(self.prod), 'SOURCE_DATE_EPOCH': '1'}

    def collect(self, success=True):
        result = subprocess.run([sys.executable, '-I', str(self.root / 'tools/ci/flash_licenses.py'),
                                 'collect', *map(str, self.sources), str(self.prod)], capture_output=True)
        self.assertEqual(result.returncode == 0, success, result.stderr.decode())

    def package(self, success=True):
        result = subprocess.run(['bash', str(self.root / 'tools/ci/package.sh'), 'flash-fixture',
                                 'test', str(self.dist), 'server', 'flash', str(self.target.parent),
                                 'oxe-server'], env=self.env, capture_output=True)
        self.assertEqual(result.returncode == 0, success, result.stderr.decode())
        archives = list(self.dist.glob('*.tar.gz')) if self.dist.exists() else []
        self.assertEqual(bool(archives), success)
        return archives

    def test_archive_carries_exact_both_upstream_texts_and_attribution(self):
        archive, = self.package()
        prefix = 'flash-fixture-test-linux-aarch64/'
        with tarfile.open(archive) as tar:
            for name, raw in self.expected.items():
                self.assertEqual(tar.extractfile(prefix + 'licenses/' + name + '-LICENSE').read(), raw)
            notices = tar.extractfile(prefix + 'THIRD_PARTY_NOTICES').read()
            self.assertIn(b'930b2c9ebd8c2cface6566bd9f11075ac07adfe0', notices)
            self.assertIn(b'7ac98f4a8089a36c0581b7c6e8973417a8fa175c', notices)
            metadata = json.load(tar.extractfile(prefix + 'FLASH-SOURCE-LICENSES.json'))
            for name, pin in PINS.items():
                self.assertEqual(metadata['sources'][name]['sha256'], pin)
                self.assertIsNone(metadata['sources'][name]['notice'])

    def test_missing_bigmoe_license_fails_closed(self):
        (self.prod / 'LICENSE-BigMoeOnEdge').unlink(); self.package(False)

    def test_tampered_llama_license_fails_closed(self):
        (self.prod / 'LICENSE-llama.cpp').write_text('changed\n'); self.package(False)

    def test_missing_source_metadata_fails_closed(self):
        (self.prod / 'FLASH-SOURCE-LICENSES.json').unlink(); self.package(False)

    def test_changed_source_revision_fails_closed(self):
        path = self.prod / 'FLASH-SOURCE-LICENSES.json'
        metadata = json.loads(path.read_text())
        metadata['sources']['BigMoeOnEdge']['revision'] = 'unreviewed'
        path.write_text(json.dumps(metadata)); self.package(False)

    def test_flash_without_prod_bundle_fails_closed(self):
        self.env.pop('OXF_GGML_PROD'); self.package(False)

    def test_unreviewed_upstream_notice_fails_closed(self):
        (self.sources[0] / 'NOTICE').write_text('new upstream notice requires reviewed pins\n')
        self.collect(False)


if __name__ == '__main__':
    unittest.main()
