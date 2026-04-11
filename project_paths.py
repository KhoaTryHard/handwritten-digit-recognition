from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def project_file(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def project_path(*parts: str) -> str:
    return str(project_file(*parts))
