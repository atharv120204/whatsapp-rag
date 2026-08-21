# Setting this up on a new Windows PC

A complete walkthrough, assuming no programming knowledge. Every command can be
copied and pasted exactly as written — nothing needs editing, as long as you put
the folder where step 2 says.

**Time needed:** about 20 minutes, most of it waiting for downloads.
**Cost:** nothing. Both AI services used here have free tiers.

---

## Contents

1. [Install the two programs it needs](#step-1--install-the-two-programs-it-needs)
2. [Download the project from GitHub](#step-2--download-the-project-from-github)
3. [Get your two free API keys](#step-3--get-your-two-free-api-keys)
4. [One-time setup](#step-4--one-time-setup)
5. [Starting the app (every time)](#step-5--starting-the-app-every-time)
6. [Enter your keys in the app](#step-6--enter-your-keys-in-the-app)
7. [Get your WhatsApp chat onto the PC](#step-7--get-your-whatsapp-chat-onto-the-pc)
8. [Load the chat](#step-8--load-the-chat)
9. [Ask it things](#step-9--ask-it-things)
10. [Shutting down](#step-10--shutting-down)
11. [When something goes wrong](#when-something-goes-wrong)

---

## Step 1 — Install the two programs it needs

### 1a. Python

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **Download Python** button.
3. Open the file that downloads.
4. **IMPORTANT:** on the first screen, tick the box at the bottom that says
   **"Add python.exe to PATH"**. It is easy to miss, and if you skip it nothing
   later in this guide will work.
5. Click **Install Now** and wait.
6. Click **Close**.

> Any version 3.12 or newer is fine. This project is tested on 3.14.

### 1b. Node.js

1. Go to **https://nodejs.org/**
2. Download the version labelled **LTS** (it means "long-term support" — the
   stable one).
3. Open the downloaded file and click **Next** through the installer, accepting
   the defaults. Do not change anything.
4. Wait for it to finish, then click **Finish**.

### 1c. Check both installed properly

1. Press the **Windows key**, type `powershell`, and press **Enter**. A dark
   blue or black window with white text opens. This is where you type commands.
2. Copy and paste this, then press Enter:

```powershell
py --version
```

You should see something like `Python 3.14.6`.

3. Now paste this and press Enter:

```powershell
node --version
```

You should see something like `v24.16.0`.

**If either one says "not recognized"** — the install did not finish, or you
missed the "Add python.exe to PATH" tick box. Restart the PC and try the check
again. If it still fails, re-run that installer.

Leave this window open; you will use it in step 4.

---

## Step 2 — Download the project from GitHub

You do **not** need to install Git or know anything about it.

1. Go to **https://github.com/atharv120204/whatsapp-rag**
2. Find the green **`< > Code`** button near the top right and click it.
3. In the small menu that opens, click **Download ZIP**.
4. The file `whatsapp-rag-main.zip` lands in your **Downloads** folder.
5. Open your Downloads folder, **right-click** that ZIP file, and choose
   **Extract All…**
6. In the box that appears, delete whatever is written there and type exactly:

```
C:\
```

7. Click **Extract**. Wait for it to finish.
8. Open **This PC → Local Disk (C:)**. You will see a folder called
   **`whatsapp-rag-main`**.
9. **Right-click that folder, choose Rename, and change it to exactly:**

```
whatsapp-rag
```

You should now have a folder at **`C:\whatsapp-rag`**, and inside it you should
see `backend`, `frontend`, `README.md` and some other files.

> **This exact location matters.** Every command below assumes
> `C:\whatsapp-rag`. If you put it somewhere else, the commands will fail with
> "cannot find path".

---

## Step 3 — Get your two free API keys

An API key is a long password that lets the app use an AI service. You need two,
both free. **Treat them like passwords — do not post them anywhere or send them
to anyone.**

### 3a. Google Gemini key — for search and reading photos

1. Go to **https://aistudio.google.com/apikey**
2. Sign in with a Google account.
3. Click **Create API key**.
4. Click the **copy** icon next to the key it shows you.
5. Paste it somewhere safe for a minute — Notepad is fine. It starts with `AQ.`

### 3b. Groq key — for answering questions

1. Go to **https://console.groq.com/keys**
2. Sign up (a Google account works).
3. Click **Create API Key**, give it any name, and click **Submit**.
4. Copy the key **now** — Groq will not show it to you again. It starts with
   `gsk_`.
5. Paste it into the same Notepad window.

> **Why two?** Groq answers questions quickly and generously on its free tier.
> Google's Gemini is the one that can build the search index and look at photos.
> Each is used for what it is good at.

---

## Step 4 — One-time setup

This installs the app's internal parts. You only ever do this **once**.

Go back to your PowerShell window (or open a new one: Windows key → type
`powershell` → Enter).

Paste each of the following **one at a time**, pressing Enter after each and
waiting for it to finish before pasting the next.

**4a. Go to the backend folder:**

```powershell
Set-Location "C:\whatsapp-rag\backend"
```

Nothing visible happens. That is correct.

**4b. Create a private space for the Python parts:**

```powershell
py -m venv .venv
```

Takes 10–30 seconds. No output means it worked.

**4c. Install the Python parts** (this one downloads a fair amount — give it a
minute or two):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You will see a lot of scrolling text. At the end you want a line starting with
**`Successfully installed`**. A yellow warning about upgrading pip is normal and
can be ignored.

**4d. Install the website parts:**

```powershell
npm install --prefix "C:\whatsapp-rag\frontend"
```

More scrolling text, then something like `added 109 packages`. Warnings in
yellow are normal.

Setup is done. You never repeat step 4.

---

## Step 5 — Starting the app (every time)

The app is two halves that both need to be running: the **engine** and the
**website**. So you need **two PowerShell windows open at once**. This feels odd
the first time; it is normal.

### Window 1 — the engine

Open PowerShell (Windows key → `powershell` → Enter) and paste these two, one at
a time:

```powershell
Set-Location "C:\whatsapp-rag\backend"
```

```powershell
.\.venv\Scripts\python.exe -m app.cli serve
```

Wait until you see a line containing:

```
Uvicorn running on http://127.0.0.1:8000
```

**Leave this window open.** Closing it switches the app off. It will look like it
has frozen — it has not; it is waiting to do work.

### Window 2 — the website

Open a **second, separate** PowerShell window. Do not reuse the first one.

```powershell
npm run dev --prefix "C:\whatsapp-rag\frontend"
```

Wait for:

```
Local:   http://localhost:5173/
```

**Leave this window open too.**

### Open it

Open your web browser and go to:

```
http://localhost:5173
```

The app appears.

> **Do not** open `http://localhost:8000`. That is the engine, not the app, and
> it will show a bare "Not Found" page. `5173` is the one you want.

---

## Step 6 — Enter your keys in the app

1. In the app, click the **Settings** tab at the top.

**The Google key:**

2. Find the box labelled **Gemini API key**.
3. Paste the key that starts with `AQ.` into it.
4. Click **Save key** (the green button next to it).

**The Groq key:**

5. Find the dropdown labelled **Chat provider** and change it to **Groq**. The
   **Model** box next to it fills in automatically — leave whatever it puts
   there.
6. A box labelled **API key** appears. Paste the key starting with `gsk_` into
   it.
7. Click **Save**.

**Check they work:**

8. Click either **Test** button. After a few seconds it lists the models your
   keys can actually use. If you see a list, both keys are good. If you see a
   red error, the key was pasted wrong — copy it again, making sure you did not
   pick up a space at either end.

You only do this once per PC. The keys are stored on this computer only, in
`C:\whatsapp-rag\data\config.json`, and are never sent anywhere except to
Google and Groq when answering your questions.

---

## Step 7 — Get your WhatsApp chat onto the PC

**On your phone:**

1. Open WhatsApp and open the chat you want.
2. Tap the chat name at the top to open its info page.
3. Scroll to the bottom and tap **Export chat**.
   - On Android it may be under the **⋮** menu → **More** → **Export chat**.
4. Choose **Attach media** if you want photos, voice notes and videos included.
   Choose **Without media** for a much smaller file that still contains every
   message.
5. Choose how to send it to yourself — **Email** is easiest, or **WhatsApp** to
   your own number, or Google Drive.

**On the PC:**

6. Download that file. It will be a `.zip` (with media) or a `.txt` (without).
7. Remember where it saved — usually **Downloads**.

> **Which should you pick?** "Attach media" is capped by WhatsApp at roughly the
> last 10,000 messages. "Without media" covers the entire history. If you want
> both the full history *and* the photos, export twice and load both files — the
> app merges them without duplicating anything.

---

## Step 8 — Load the chat

1. In the app, click the **Add chat** tab.
2. In the **Load into** dropdown, leave it on **+ A new archive**.
3. Type a name in the box next to it — whatever helps you recognise it, e.g.
   `Family` or `College Group`.
4. **Recommended for your first load:** untick the box that says **"Describe
   photos, voice notes and video"**. Leave **"Build semantic search index"**
   ticked.
   - Why: describing photos is slow on the free tier — Google allows about 20
     photos a day, so a chat with 400 photos would take weeks. Everything else
     works immediately without it, and the app will offer to do the photos later,
     a day at a time, telling you the cost first.
5. Drag your `.zip` or `.txt` file onto the dotted box, or click it and browse.
6. A progress bar appears. A large chat takes a few minutes. **Do not close the
   PowerShell windows while it runs.**
7. When it finishes you will see a summary: how many messages, how many people,
   the date range.

---

## Step 9 — Ask it things

Click the **Ask** tab and type a question.

**Questions that come back in a couple of seconds** — these are counted directly
from the data:

- How many messages did each person send?
- Who starts conversations most often?
- What time of day is this group most active?
- Who replies the fastest?

**Questions that take one to two minutes** — these have to search and read
through the conversation before answering:

- What did we decide about the trip?
- Why was everyone annoyed in March?
- Summarise the argument about the bill

**The slow ones are not broken.** The free AI service allows only a limited
amount of text per minute, so the app has to pause between steps. Let it run.

Under every answer there is a line showing exactly which query produced it —
click it. If the app tells you a number, you can check where the number came
from. That is the whole point of it.

Also worth exploring: **Dashboard** (charts), **Insights** (funniest and most
heated conversations), **Media** (searchable photos and voice notes), **Browse**
(read the raw messages).

---

## Step 10 — Shutting down

1. Close the browser tab.
2. Click each PowerShell window and press **Ctrl + C**, then close it.

Your data stays on the PC. Next time, start again from **Step 5** — steps 1
through 4 are never repeated.

---

## When something goes wrong

### "Internal Server Error" in the app

The engine is not running. Its PowerShell window was closed, or it never
started. Redo **Step 5, Window 1**, then refresh the browser.

This is by far the most common problem.

### `... is not recognized as the name of a cmdlet`

You are in the wrong folder. Run the `Set-Location` command first, then the one
that failed. Every command block above that needs a folder is preceded by its
`Set-Location`.

If `py` itself is not recognised, Python was installed without the "Add
python.exe to PATH" tick box. Re-run the Python installer and make sure it is
ticked.

### `only one usage of each socket address` / `address already in use`

The engine is already running in another window. Either use it, or find that
window and press Ctrl + C first.

### `Could not read package.json`

You are in the wrong folder for that command. Use the full version:

```powershell
npm run dev --prefix "C:\whatsapp-rag\frontend"
```

### `Port 5173 is in use, trying another one...`

The website is already running elsewhere. It will open on `5174` instead — that
works fine, just use `http://localhost:5174`. Or close the other window.

### `Archive 'X' is open in another process`

The engine and a command are both trying to use the data at once. Stop the
engine (Ctrl + C in Window 1) before running any command other than `serve`.

### "no quota left today for …"

You have used up one AI model's free daily allowance. Go to **Settings** and
pick a different model from the list — the allowances are per model, so another
one usually still has room. Otherwise it resets tomorrow.

### The answer is wrong or mentions things not in your chat

Click the trace under the answer to see what it actually searched. If it made no
tool calls at all, that is a bug worth reporting — the app is built to refuse
answering without consulting your chat.

### Nothing above matches

Look at the two PowerShell windows. The last few lines of red text usually say
plainly what is wrong. Copy them out; they are the useful part.

---

## What is on your computer, and what leaves it

**Stays on this PC, always:**

- your messages, photos and voice notes, in `C:\whatsapp-rag\data`
- your API keys, in `C:\whatsapp-rag\data\config.json`

**Leaves the PC:** when you ask a question, the relevant excerpts of your chat
are sent to Groq and Google so they can answer it. When you ask for photos to be
described, those photos are sent to Google. Nothing else is transmitted, there
is no account, and no server belonging to this project ever sees your data.

**To delete everything:** delete the `C:\whatsapp-rag` folder. That is all of
it.
