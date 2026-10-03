# Connecting Gmail to Autobot (one time, about 10 minutes)

Autobot reads your mail and creates drafts through Google's official Gmail API. It never sees
your Google password, and it sends an email only when you approve that specific email in the
butler inbox.

What you'll end up with: two files in `C:\Users\User 1\.autobot\`:
`google_client_secret.json` (identifies "your Autobot" to Google) and `gmail_token.json`
(your sign-in, written automatically in step 7).

## 1. Create a Google Cloud project
1. Open <https://console.cloud.google.com/> and sign in with the Gmail account Autobot should use.
2. Click the project picker at the top → **New Project** → name it `Autobot` → **Create**.
   Make sure `Autobot` is the selected project afterwards.

## 2. Turn on the Gmail API
Open <https://console.cloud.google.com/apis/enableflow;apiid=gmail.googleapis.com>, confirm the
`Autobot` project is selected, and click **Enable**.

## 3. Set up the consent screen
1. Menu (☰) → **Google Auth platform** → **Branding** → **Get started**.
2. App name: `Autobot`. User support email: your address. **Next**.
3. Audience: **External** (a personal Gmail account can't use "Internal"). **Next**.
4. Contact information: your address. **Next** → accept the policy → **Create**.

## 4. Add yourself as a test user
Menu → **Google Auth platform** → **Audience** → under **Test users** click **Add users** →
enter your Gmail address → **Save**.

## 5. Create the desktop client and download it
1. Menu → **Google Auth platform** → **Clients** → **Create client**.
2. Application type: **Desktop app**. Name: `Autobot desktop`. **Create**.
3. Click **Download JSON** in the dialog.
4. Rename the file to `google_client_secret.json` and move it to `C:\Users\User 1\.autobot\`.

## 6. Install the Google libraries (once)
In the Autobot folder:

    pip install google-api-python-client google-auth-oauthlib

## 7. Sign in
    autobot butler gmail-auth

A browser tab opens. Choose your account. Google will say the app isn't verified; that's
expected for a personal project. Click **Continue**, allow the two permissions (read mail;
manage drafts and send), and close the tab when it says so. The command prints
`Gmail connected for you@gmail.com`.

## 8. Start using it
    autobot butler add email "Track my email: tell me what needs me and draft replies where a reply from me is clearly needed." --mode triage --every 120

Every two hours the butler pulls mail from the last two days (skipping Promotions and Social),
writes a digest, and turns reply drafts into real Gmail drafts. Each draft shows up in your inbox
(`autobot butler inbox`, or the phone page) as **Send / Don't send**. "Don't send" leaves it in
Gmail Drafts so you can edit and send it yourself.

For new emails (cold outreach, personal notes):

    autobot butler add email "Draft a short, specific email to <person/company> about <purpose>. Facts to use: ..." --mode compose

Optional: put notes on how you write emails (sign-off, tone, things you never say) in
`C:\Users\User 1\.autobot\email_style.md`. Every email task reads it.

## Things to know
- **The weekly re-sign-in.** While your consent screen is in "Testing" status, Google expires the
  sign-in after about 7 days. When that happens, the butler puts a message in your inbox asking
  you to run `autobot butler gmail-auth` again. To stop this, go to **Google Auth platform →
  Audience** and click **Publish app**. Google keeps showing the "unverified app" warning at
  sign-in, which is fine for your own use.
- **Permissions requested:** `gmail.readonly` and `gmail.compose` only. Autobot has no permission
  to delete mail or change labels.
- **Privacy:** triage sends the text of your recent emails to Claude through your Claude Code
  subscription, because that's the worker that reads them. The triage worker has no shell and
  no web access, so an email can't make it run commands or send data anywhere.
- **Disconnect** at any time: delete `gmail_token.json`, or remove Autobot at
  <https://myaccount.google.com/permissions>.
