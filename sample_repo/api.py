
from auth import AuthService
from db import save_audit_log

auth_service = AuthService()

def login_endpoint(request):
    username = request["username"]
    password = request["password"]
    result = auth_service.login_user(username, password)
    save_audit_log({"action": "login", "user": username})
    return result

def logout_endpoint(request):
    token = request["token"]
    return auth_service.logout_user(token)
