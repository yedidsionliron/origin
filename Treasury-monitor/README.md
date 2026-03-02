# Treasury Monitor

Checks the U.S. 10-Year Treasury yield daily and sends an email alert when it drops below a configured threshold.

**Data source:** Yahoo Finance (`^TNX` ticker, no API key needed)
**Email:** Gmail SMTP with App Password

---

## Files

```
Treasury-monitor/
├── treasury_monitor.py   # Main script
├── config.yaml           # Threshold + email settings
├── test_treasury_monitor.py
└── README.md
```

---

## First-Time Setup

### 1. Install Python dependencies

```powershell
pip install yfinance pyyaml
```

### 2. Edit config.yaml

Set your threshold and email address:

```yaml
treasury:
  threshold: 3.75   # Alert when yield drops below this %
  ticker: "^TNX"

email:
  sender: "you@gmail.com"
  recipient: "you@gmail.com"
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
```

### 3. Create a Gmail App Password

1. Go to **myaccount.google.com/apppasswords**
2. Name it (e.g. "Treasury Monitor") → Create
3. Copy the 16-character password (no spaces)

> Requires Gmail 2-Step Verification to be enabled.

### 4. Save the password as a permanent environment variable

Open PowerShell **as Administrator** (`Win + R` → `powershell` → `Ctrl + Shift + Enter`):

```powershell
[System.Environment]::SetEnvironmentVariable("SMTP_PASSWORD", "your16charpassword", "User")
```

### 5. Test it manually

In a regular PowerShell terminal:

```powershell
$env:SMTP_PASSWORD = "your16charpassword"
cd "C:\Users\<you>\Downloads\Python codes\Treasury-monitor"
python treasury_monitor.py
```

Expected output (yield above threshold):
```
Current 10Y Treasury yield: 4.21%
Yield 4.21% >= threshold 3.75%. No alert sent.
```

To test the email, temporarily set `threshold: 5.5` in `config.yaml`, run again, then revert.

---

## Schedule Daily (Windows Task Scheduler)

Open PowerShell **as Administrator** and run:

```powershell
$action = New-ScheduledTaskAction `
  -Execute "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" `
  -Argument "treasury_monitor.py" `
  -WorkingDirectory "C:\Users\<you>\Downloads\Python codes\Treasury-monitor"

$trigger = New-ScheduledTaskTrigger -Daily -At "08:00AM"

Register-ScheduledTask -TaskName "Treasury Monitor" -Action $action -Trigger $trigger -RunLevel Highest
```

Replace `<you>` with your Windows username. Change `08:00AM` to your preferred time.

To verify: right-click the task in Task Scheduler → **Run** — check for output or email.

---

## Updating the Threshold

Just edit `config.yaml` and save. The script reads it fresh on every run — no need to touch the scheduled task.

---

## Moving to a New Computer

1. Copy the `Treasury-monitor/` folder
2. Install Python + `pip install yfinance pyyaml`
3. Repeat steps 4–5 above (the App Password is tied to your Gmail, not the machine)
4. Re-run the Task Scheduler PowerShell command

---

## Running Tests

```powershell
python -m pytest test_treasury_monitor.py -v
```

All 5 tests should pass.
