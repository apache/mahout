#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import subprocess
import sys
from pathlib import Path

import pytest


class TestTypeChecking:
    """Test class for validating type hints using mypy."""

    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture
    def qumat_dir(self, project_root):
        """Get the qumat package directory."""
        return project_root / "qumat"

    def run_mypy(self, target_path: Path) -> tuple[int, str, str]:
        """Run mypy on the specified path and return results.

        Args:
            target_path: Path to the file or directory to type check.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        result = subprocess.run(
            [sys.executable, "-m", "mypy", str(target_path)],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr

    def test_qiskit_backend_types(self, qumat_dir):
        """Test that qiskit_backend.py passes mypy type checking."""
        qiskit_backend = qumat_dir / "qiskit_backend.py"
        assert qiskit_backend.exists(), "qiskit_backend.py not found"

        returncode, stdout, stderr = self.run_mypy(qiskit_backend)

        # Check for success or only notes/warnings (not errors)
        assert returncode == 0, (
            f"Type checking failed for qiskit_backend.py:\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    def test_cirq_backend_types(self, qumat_dir):
        """Test that cirq_backend.py passes mypy type checking."""
        cirq_backend = qumat_dir / "cirq_backend.py"
        assert cirq_backend.exists(), "cirq_backend.py not found"

        returncode, stdout, stderr = self.run_mypy(cirq_backend)

        assert returncode == 0, (
            f"Type checking failed for cirq_backend.py:\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    def test_amazon_braket_backend_types(self, qumat_dir):
        """Test that amazon_braket_backend.py passes mypy type checking."""
        amazon_braket_backend = qumat_dir / "amazon_braket_backend.py"
        assert amazon_braket_backend.exists(), "amazon_braket_backend.py not found"

        returncode, stdout, stderr = self.run_mypy(amazon_braket_backend)

        assert returncode == 0, (
            f"Type checking failed for amazon_braket_backend.py:\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    def test_all_backend_modules_together(self, qumat_dir):
        """Test that all backend modules pass type checking together.

        This helps catch issues with type consistency across modules.
        """
        returncode, stdout, stderr = self.run_mypy(qumat_dir)

        assert returncode == 0, (
            f"Type checking failed for qumat package:\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    def test_union_type_syntax(self, qumat_dir):
        """Test that modern union type syntax (int | None) is supported."""
        qiskit_backend = qumat_dir / "qiskit_backend.py"

        # Read the file and check for modern union syntax
        content = qiskit_backend.read_text()

        # Verify modern union syntax is present
        assert "int | None" in content or "float | str" in content, (
            "Expected to find modern union type syntax (e.g., 'int | None')"
        )

        # Ensure mypy can handle it
        returncode, stdout, stderr = self.run_mypy(qiskit_backend)
        assert returncode == 0, (
            f"Modern union syntax caused type checking errors:\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    def test_no_type_ignore_comments(self, qumat_dir):
        """Verify that backend modules don't rely on type: ignore comments.

        This ensures type hints are genuinely correct rather than suppressed.
        """
        backend_files = [
            "qiskit_backend.py",
            "cirq_backend.py",
            "amazon_braket_backend.py",
        ]

        for filename in backend_files:
            filepath = qumat_dir / filename
            content = filepath.read_text()

            # Count type: ignore comments
            ignore_count = content.count("# type: ignore")

            # We want minimal to no type: ignore comments
            assert ignore_count == 0, (
                f"{filename} contains {ignore_count} '# type: ignore' comment(s). "
                f"Type hints should be correct without suppression."
            )
