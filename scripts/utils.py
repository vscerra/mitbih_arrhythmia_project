import os

def get_project_path(*subpaths):
    """
    Returns an absolute path from the project root, optionally joined with one or more subpaths
    """

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root, *subpaths)