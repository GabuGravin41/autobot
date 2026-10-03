from pathlib import Path
import json
import webbrowser

secret_path = Path(r"C:\Users\User 1\.autobot\google_client_secret.json")
token_path = Path(r"C:\Users\User 1\.autobot\gmail_token.json")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

def main():
    from google_auth_oauthlib.flow import InstalledAppFlow
    print("Loading secret from:", secret_path)
    flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
    print("=" * 60)
    print("OAUTH AUTHORIZATION URL:")
    print(auth_url)
    print("=" * 60)
    
    # Try running local server with short timeout or open browser
    try:
        creds = flow.run_local_server(port=0, open_browser=True, timeout_seconds=120)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        print("SUCCESS: gmail_token.json written!")
        from googleapiclient.discovery import build
        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
        email = svc.users().getProfile(userId="me").execute().get("emailAddress", "")
        print("Connected email:", email)
    except Exception as e:
        print("Flow error or timeout:", e)

if __name__ == "__main__":
    main()
