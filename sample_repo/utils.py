
def validate_request_fields(request, required_fields):
    for field in required_fields:
        if field not in request:
            raise KeyError(field)
    return True

def format_response(data):
    return {"data": data, "ok": True}
