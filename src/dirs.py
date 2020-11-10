import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
HOME = os.getenv('PROJECT_HOME')
DATAHOME = Path(os.path.join(os.getenv('PROJECT_HOME'), 'data'))
MODELHOME = Path(os.path.join(os.getenv('PROJECT_HOME'), 'models'))
