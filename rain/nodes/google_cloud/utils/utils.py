from google.oauth2 import service_account

from dotenv import load_dotenv
from requests import get
from os import getenv


def get_gcp_credentials():
    load_dotenv()
    
    try:
        if getenv("SERVICE_ACCOUNT_FILE"):
            response = get(getenv("SERVICE_ACCOUNT_FILE"))
            if response.status_code == 200:
                json_data = response.json()
                credentials = service_account.Credentials.from_service_account_info(json_data)
        else:
            raise Exception("SERVICE_ACCOUNT_FILE environment variable is not set")
    except:
            raise ValueError("Invalid JSON format")
    return credentials