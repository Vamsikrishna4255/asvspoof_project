import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==============================
# CONFIGURATION
# ==============================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "krishnavamsi0842@gmail.com"       # Your Gmail
APP_PASSWORD = "paeg xkkj nsfc evzl"           # 16-char Gmail App Password (NO SPACES)

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
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENTS, message.as_string())
        server.quit()
        print("✅ Spoof alert email sent successfully!")

    except Exception as e:
        print("❌ Failed to send email:", e)
