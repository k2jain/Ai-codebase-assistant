
def connect_db():
    return "connected"

def get_user_by_id(user_id):
    return {"id": user_id, "username": "krrish", "role": "admin"}

def save_audit_log(event):
    return {"saved": True, "event": event}
