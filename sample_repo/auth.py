
class AuthService:
    def __init__(self):
        self.valid_users = {"admin": "secret", "krrish": "mlrocks"}

    def login_user(self, username, password):
        if username in self.valid_users and self.valid_users[username] == password:
            return {"status": "success", "token": f"token_for_{username}"}
        return {"status": "failure", "reason": "invalid credentials"}

    def logout_user(self, token):
        return {"status": "logged_out", "token": token}
