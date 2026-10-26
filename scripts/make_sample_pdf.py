"""Generates a fictional company handbook PDF for demos and the evaluation set."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("Refund Policy", [
        "Customers may request a full refund within 30 days of purchase if the product is unused and in its original packaging.",
        "Refunds are issued to the original payment method within 5 to 7 business days after we receive the returned item.",
        "Digital products and gift cards are non-refundable. Shipping charges are not refunded unless the item arrived damaged.",
    ]),
    ("Warranty", [
        "All AcmeCo products come with a 2-year limited warranty starting from the date of purchase.",
        "The warranty covers manufacturing defects. It does not cover accidental damage, water damage, or unauthorized repairs.",
        "To claim warranty service, contact support with your order number and proof of purchase.",
    ]),
    ("Registration", [
        "To register your product, create an account on the AcmeCo portal and enter the serial number printed on the back of the device.",
        "The following documents are required for registration: a valid government photo ID, the original invoice, and the product serial number.",
        "Business customers must additionally provide a GST registration certificate.",
    ]),
    ("Troubleshooting", [
        "If the device does not power on, hold the power button for 15 seconds, then connect the charger and wait 10 minutes before trying again.",
        "If the Wi-Fi connection keeps dropping, restart the router, forget the network on the device, and reconnect with the password.",
        "For a factory reset, press and hold the reset pin for 10 seconds until the status light blinks red. This erases all personal data.",
    ]),
]

if __name__ == "__main__":
    out = Path(__file__).parent.parent / "sample_docs" / "AcmeCo_Support_Handbook.pdf"
    out.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        for p in paras:
            story += [Paragraph(p, styles["Normal"]), Spacer(1, 10)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=A4).build(story)
    print("written", out)
