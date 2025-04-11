# analysis/visualization/superset_connector.py
import requests

class SupersetConnector:
    def __init__(self, base_url, username, password):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.authenticate(username, password)

    def authenticate(self, username, password):
        login_url = f"{self.base_url}/api/v1/security/login"
        payload = {
            "username": username,
            "password": password,
            "provider": "db",
            "refresh": True
        }
        response = self.session.post(login_url, json=payload)
        if response.status_code == 200:
            access_token = response.json().get("access_token")
            if not access_token:
                raise Exception("No access token returned. Check credentials.")
            self.session.headers.update({"Authorization": f"Bearer {access_token}"})
            print("Authenticated with Superset.")
        else:
            raise Exception(f"Failed to authenticate with Superset: {response.text}")

    def run_query(self, sql, schema):
        query_url = f"{self.base_url}/api/v1/query/"
        payload = {"query": sql, "schema": schema}
        response = self.session.post(query_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed: {response.text}")

    def trigger_refresh(self, dashboard_id):
        refresh_url = f"{self.base_url}/api/v1/dashboard/{dashboard_id}/refresh"
        response = self.session.post(refresh_url)
        if response.status_code == 200:
            print("Dashboard refresh triggered.")
        else:
            raise Exception(f"Dashboard refresh failed: {response.text}")

if __name__ == "__main__":
    BASE_URL = "http://localhost:8088"
    USERNAME = "admin"
    PASSWORD = "admin"
    try:
        connector = SupersetConnector(BASE_URL, USERNAME, PASSWORD)
        result = connector.run_query("SELECT * FROM log_table LIMIT 10;", "public")
        print("Query result:", result)
        connector.trigger_refresh(1)
    except Exception as e:
        print("Error:", e)
