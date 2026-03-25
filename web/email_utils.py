import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==============================
# CONFIGURATION
# ==============================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "krishnavamsi0842@gmail.com"       # Your Gmail
APP_PASSWORD = "yofe cgoe mdqf pebg"           # 16-char Gmail App Password (NO SPACES)

RECIPIENTS = [
    "krishnavamsi0842@gmail.com",
]

# ==============================
# EMAIL FUNCTION
# ==============================
def send_spoof_alert(spoof_confidence):
    subject = "⚠️ Spoof Audio Detected"
    body = f"""
Hello,

A spoofed audio clip has been detected by the system.

Spoof Confidence: {spoof_confidence}

Please take necessary action.

"""

    # Create email
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = ", ".join(RECIPIENTS)
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    # Send email
    try:
        print(f"[DEBUG] Connecting to {SMTP_SERVER}:{SMTP_PORT}...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        print(f"[DEBUG] Connected. Starting TLS...")
        server.starttls()
        print(f"[DEBUG] TLS started. Logging in as {SENDER_EMAIL}...")
        server.login(SENDER_EMAIL, APP_PASSWORD)
        print(f"[DEBUG] Login successful. Sending email to {RECIPIENTS}...")
        server.sendmail(SENDER_EMAIL, RECIPIENTS, message.as_string())
        server.quit()
        print("✅ Spoof alert email sent successfully!")

    except Exception as e:
        import traceback
        print("❌ Failed to send email:", e)
        traceback.print_exc()
