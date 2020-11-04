import os
from dotenv import load_dotenv

load_dotenv()
HOME = os.getenv('PROJECT_HOME')
DATAHOME = os.path.join(os.getenv('PROJECT_HOME'), 'data')
