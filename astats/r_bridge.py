import subprocess
import tempfile
import os
import re


def run_r_script(code: str) -> str:
    """
    Executes an R script safely via subprocess.
    Writes the script to a temp file, executes via 'Rscript', captures output.

    Smart detection: if stderr contains a 'no package called' message,
    returns a structured error so the LLM can self-correct with install.packages().
    """
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".R")
        with os.fdopen(fd, "w") as f:
            f.write(code)

        result = subprocess.run(
            ["Rscript", path],
            capture_output=True,
            text=True,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # ── Smart missing-package detection ───────────────────────────────
        # Look for R's canonical error pattern: no package called 'xyz'
        missing_pkg_match = re.search(
            r"there is no package called ['\"]([^'\"]+)['\"]",
            stderr,
            re.IGNORECASE,
        )
        if missing_pkg_match:
            pkg_name = missing_pkg_match.group(1)
            return (
                f"Error: R package '{pkg_name}' is not installed.\n"
                f"To fix this, first run: install.packages('{pkg_name}', repos='https://cloud.r-project.org')\n"
                f"Then re-run your original analysis code.\n"
                f"[Raw stderr]: {stderr.strip()}"
            )

        if result.returncode != 0:
            return (
                f"Error executing R script:\n"
                f"Return Code: {result.returncode}\n"
                f"Output: {stdout}\n"
                f"Stderr: {stderr}"
            )

        output = stdout
        if stderr.strip():
            output += f"\n[STDERR Warn]:\n{stderr}"
        return output

    except FileNotFoundError:
        return (
            "Error: 'Rscript' executable not found. "
            "Please install R and ensure it is in your system PATH."
        )
    except Exception as e:
        return f"Unexpected Error: {e}"
    finally:
        if path and os.path.exists(path):
            os.remove(path)


def run_python_code(code: str) -> str:
    """
    Executes Python code in a sandboxed exec() context.
    pandas (pd) and numpy (np) are pre-imported into the exec globals.

    Warning: This is a prototype execution environment.
    Do not use in production without a proper sandbox (e.g. Docker / RestrictedPython).
    """
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec_globals = {
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
        }
        exec(code, exec_globals)
        return sys.stdout.getvalue()
    except Exception:
        import traceback
        return f"Error executing Python:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
