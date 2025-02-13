from fpdf import FPDF

# Generate PDF report function
def generate_pdf_report(detections):
    """Generate a PDF report from detections."""
    if not detections:
        st.warning("No damage detected. Report generation skipped.")
        return None
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Road Damage Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for det in detections:
        pdf.cell(200, 10, f"Crack Type: {det.label}", ln=True)
        pdf.cell(200, 10, f"Confidence Score: {round(det.score, 2)}", ln=True)
        pdf.cell(200, 10, f"Bounding Box: {det.box.tolist()}", ln=True)
        pdf.ln(5)
    
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

# Add a button to download the report
if detections:
    report_pdf = generate_pdf_report(detections)
    if report_pdf:
        st.download_button(
            label="Download Detection Report",
            data=report_pdf,
            file_name="RDD_Detection_Report.pdf",
            mime="application/pdf"
        )
