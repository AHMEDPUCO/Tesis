SCENARIOS = [
    {
        "name": "auth_anomaly_valid_account",
        "technique_ids": ["T1078"],
        "description": "Login exitoso inusual + actividad posterior"
    },
    {
        "name": "auth_bruteforce_then_success",
        "technique_ids": ["T1110"],
        "description": "Varios intentos fallidos + 1 éxito"
    },
    {
        "name": "lateral_like_remote_service",
        "technique_ids": ["T1021"],
        "description": "Conexión remota sospechosa entre hosts"
    },
]
