import pandas as pd

# تحميل الملف الأصلي
df = pd.read_csv("promise_data.csv")

# عرض الأعمدة للتأكد
print("📄 الأعمدة المتوفرة:", list(df.columns))

# نعيد تسمية الأعمدة المطلوبة
df = df.rename(columns={
    "RequirementText": "requirement",
    "IsFunctional": "label"
})

# نحول العمود label لقيم نصية مفهومة
df["label"] = df["label"].astype(str).str.strip().str.lower().map({
    "1": "functional",       # لو الملف بيستخدم 1 و 0
    "true": "functional",
    "yes": "functional",
    "functional": "functional",
    "0": "non-functional",
    "false": "non-functional",
    "no": "non-functional",
    "non-functional": "non-functional"
})

# إزالة الصفوف غير المفهومة
df = df[df["label"].isin(["functional", "non-functional"])]

# حفظ النتيجة
df.to_csv("promise_data_cleaned.csv", index=False)
print("✅ File saved as promise_data_cleaned.csv with", len(df), "rows.")
