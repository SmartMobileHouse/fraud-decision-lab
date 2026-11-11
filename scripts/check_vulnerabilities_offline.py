"""Run an already installed pip-audit with network blocked; cache misses are not a pass."""

from __future__ import annotations

import importlib.util
import runpy
import socket
import sys


def main() -> int:
    if importlib.util.find_spec("pip_audit") is None:
        print("NOT VERIFIED: pip-audit is not installed. No tool or advisory data was downloaded.")
        return 2

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise OSError("offline advisory check: network is disabled; an uncached response is NOT VERIFIED")

    socket.socket.connect = blocked  # type: ignore[method-assign, assignment]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign, assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]
    sys.argv = ["pip-audit", "--disable-pip", "--no-deps", "--progress-spinner", "off"]
    try:
        runpy.run_module("pip_audit", run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code:
            print(
                "Offline scan did not pass: review findings or missing advisory evidence; do not infer a clean result."
            )
        else:
            print("Cached advisory check completed; independently review advisory freshness before release.")
        return code
    except Exception as exc:
        print(f"NOT VERIFIED: offline advisory check could not complete ({type(exc).__name__}).")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
