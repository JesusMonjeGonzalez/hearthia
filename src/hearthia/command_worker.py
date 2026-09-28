"""Exec wrapper: apply modest resource defaults before a project command."""

import os
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 121))
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft = 512 if hard == resource.RLIM_INFINITY else min(512, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    os.nice(5)
    try:
        os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
    except OSError as exc:
        # Shell convention: 127 means "command not found". A traceback here
        # only confuses the model and the log.
        where = "not found" if isinstance(exc, FileNotFoundError) else f"cannot execute ({exc})"
        print(
            f"{sys.argv[1]}: {where}. PATH searched was:\n  "
            + os.environ.get("PATH", "").replace(":", "\n  "),
            file=sys.stderr,
        )
        raise SystemExit(127) from None


if __name__ == "__main__":
    main()
