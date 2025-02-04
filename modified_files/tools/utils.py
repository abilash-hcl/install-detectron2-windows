import os
from pathlib import Path

def get_project_dir():
    """This method returns the path for the repo
    """   
    # path = os.getcwd() 
    path = os.path.dirname(os.path.abspath(__file__))  
    return Path(path).resolve().parents[0] 
    return os.path.dirname(path)
