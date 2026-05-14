import os
from dotenv import load_dotenv

load_dotenv()

print("KEY =", os.getenv("GOOGLE_API_KEY"))