import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ===================== 1. Load dataset =====================
data = pd.read_csv("StudentsPerformance.csv")

# ===================== 2. Ensure required columns =====================
required_cols = ['student_id','attendance','study_hours','assignment_score','internal_marks','exam_score','internet_access']
for col in required_cols:
    if col not in data.columns:
        if col == 'student_id':
            data[col] = np.arange(1,len(data)+1)
        elif col == 'internet_access':
            data[col] = np.random.choice([0,1], size=len(data), p=[0.3,0.7])
        else:
            data[col] = np.random.randint(40,100,size=len(data))

# ===================== 3. Final score =====================
data['final_score'] = (data['assignment_score']*0.3 + data['internal_marks']*0.3 + data['exam_score']*0.4)

# ===================== 4. Original risk level =====================
def risk_level(score):
    if score < 65:
        return "High Risk"
    elif score < 80:
        return "Medium Risk"
    else:
        return "Low Risk"
data['risk_level'] = data['final_score'].apply(risk_level)

# ===================== 5. Encode labels for ML =====================
label_enc = LabelEncoder()
data['risk_label'] = label_enc.fit_transform(data['risk_level'])  # 0=Low,1=Medium,2=High

# ===================== 6. Features & target =====================
features = ['attendance','study_hours','assignment_score','internal_marks','exam_score','internet_access']
X = data[features]
y = data['risk_label']

# ===================== 7. Train ML model =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ===================== 8. Predict risk using ML =====================
data['predicted_label'] = clf.predict(X)
data['predicted_risk'] = label_enc.inverse_transform(data['predicted_label'])
data['risk_probability'] = clf.predict_proba(X).max(axis=1)

# ===================== 9. Explainable rules =====================
def risk_reason(row):
    reasons = []
    if row['attendance'] < 75: reasons.append("Low attendance")
    if row['study_hours'] < 2: reasons.append("Low study hours")
    if row['assignment_score'] < 50: reasons.append("Poor assignment score")
    if row['internal_marks'] < 50: reasons.append("Low internal marks")
    if row['exam_score'] < 50: reasons.append("Low exam score")
    if row['final_score'] < 65: reasons.append("Low final score")
    return ", ".join(reasons) if reasons else "None"

def early_warning(row):
    conditions = 0
    if row['attendance'] < 75: conditions += 1
    if row['study_hours'] < 2: conditions += 1
    if row['assignment_score'] < 50: conditions += 1
    if row['internal_marks'] < 50: conditions += 1
    if row['final_score'] < 65 and conditions >= 2:
        return "Early Warning"
    return "No Warning"

data['risk_reason'] = data.apply(risk_reason, axis=1)
data['early_warning'] = data.apply(early_warning, axis=1)

# ===================== 10. Graph =====================
plt.figure(figsize=(6,4))
risk_counts = data['predicted_risk'].value_counts()
colors_list = ['red','orange','green']
plt.bar(risk_counts.index, risk_counts.values, color=colors_list)
plt.title("Predicted Risk Level Distribution")
plt.ylabel("Number of Students")
plt.tight_layout()
buf = BytesIO()
plt.savefig(buf, format='PNG')
buf.seek(0)
plt.close()

# ===================== 11. PDF report =====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_name = f"EduPulse_Model_Report_{timestamp}.pdf"
doc = SimpleDocTemplate(pdf_name, pagesize=A4, rightMargin=20,leftMargin=20, topMargin=20,bottomMargin=20)
elements = []
styles = getSampleStyleSheet()

# ---------------- Cover ----------------
elements.append(Paragraph("EduPulse: AI-Powered Academic Risk & Early Warning Report", styles["Heading1"]))
elements.append(Paragraph(f"Generated on: {timestamp}", styles["BodyText"]))
elements.append(Spacer(1,12))
img = Image(buf, width=400, height=250)
elements.append(img)
elements.append(PageBreak())

# ---------------- Function to create tables with FinalScore colored ----------------
def create_risk_table(df, title):
    elements.append(Paragraph(title, styles["Heading2"]))
    table_data = [["ID","Attend%","StudyHrs","Assign","Internal","Exam","FinalScore","RiskProb","Risk","Warning","Reason"]]
    for _, row in df.iterrows():
        table_data.append([
            row['student_id'], row['attendance'], row['study_hours'], row['assignment_score'],
            row['internal_marks'], row['exam_score'], f"{row['final_score']:.2f}",
            f"{row['risk_probability']:.2f}", row['predicted_risk'], row['early_warning'], row['risk_reason']
        ])
    col_widths = [30,40,45,45,45,40,50,50,60,60,180]
    t = Table(table_data, colWidths=col_widths)
    ts = TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.black),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ALIGN',(1,1),(-2,-1),'CENTER')
    ])
    # Color FinalScore column only
    light_red = colors.Color(1,0.6,0.6)
    light_orange = colors.Color(1,0.85,0.6)
    light_green = colors.Color(0.7,1,0.7)
    for i, row in enumerate(table_data[1:], start=1):
        if row[8]=="High Risk":
            ts.add('BACKGROUND', (6,i), (6,i), light_red)
        elif row[8]=="Medium Risk":
            ts.add('BACKGROUND', (6,i), (6,i), light_orange)
        elif row[8]=="Low Risk":
            ts.add('BACKGROUND', (6,i), (6,i), light_green)
    t.setStyle(ts)
    elements.append(t)
    elements.append(PageBreak())

# ---------------- High, Medium, Low ----------------
create_risk_table(data[data['predicted_risk']=="High Risk"], "High Risk Students (Immediate Attention)")
create_risk_table(data[data['predicted_risk']=="Medium Risk"], "Medium Risk Students")
# Top 15 Low Risk Motivation
top_low_risk = data[data['predicted_risk']=="Low Risk"].sort_values('final_score', ascending=False).head(15)
create_risk_table(top_low_risk, "Top 15 Low Risk Students (Motivation)")
# Complete Table
create_risk_table(data, "Complete Student Academic Risk Report")

# ---------------- Top 5 At-Risk Students ----------------
elements.append(Paragraph("Explainable AI Insights - Top 5 At-Risk Students", styles["Heading2"]))
top5_risk = data.sort_values('risk_probability', ascending=False).head(5)
table_data = [["ID","RiskProb","Risk","Reason"]]
for _, row in top5_risk.iterrows():
    table_data.append([row['student_id'], f"{row['risk_probability']:.2f}", row['predicted_risk'], row['risk_reason']])
t = Table(table_data, colWidths=[30,50,60,250])
ts = TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
    ('GRID',(0,0),(-1,-1),0.25,colors.grey),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
])
t.setStyle(ts)
elements.append(t)
elements.append(PageBreak())

# ---------------- Recommendations ----------------
elements.append(Paragraph("Recommendations / Action Plan", styles["Heading2"]))
high_medium = data[data['predicted_risk'].isin(["High Risk","Medium Risk"])]
for _, row in high_medium.iterrows():
    recs = []
    if row['attendance'] < 75: recs.append("Increase attendance ≥ 75%")
    if row['study_hours'] < 2: recs.append("Increase study hours ≥ 2 per day")
    if row['assignment_score'] < 50: recs.append("Improve assignment scores")
    if row['internal_marks'] < 50: recs.append("Improve internal marks")
    if row['exam_score'] < 50: recs.append("Focus on exam preparation")
    if row['final_score'] < 65: recs.append("Improve overall score to ≥65%")
    
    # Fill 'None' if no recommendations
    if not recs:
        recs.append("None")
    
    elements.append(Paragraph(f"Student ID {row['student_id']}: {', '.join(recs)}", styles["BodyText"]))
elements.append(PageBreak())

# ---------------- Graphical Analysis ----------------
elements.append(Paragraph("Graphical Risk Analysis", styles["Heading2"]))
img = Image(buf, width=400, height=250)
elements.append(img)

# ---------------- Build PDF ----------------
doc.build(elements)
print(f"\n📄 PDF Generated: {pdf_name}")
print("🎓 EduPulse AI Model Analysis Complete")
