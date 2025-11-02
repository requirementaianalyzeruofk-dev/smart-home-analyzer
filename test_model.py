from transformers import pipeline

# تحميل النموذج والمُحوّل (Tokenizer) من مجلدك المحلي
classifier = pipeline(
    "text-classification",
    model="./finetuned_model",
    tokenizer="./finetuned_model",
    device=-1
)

# اختبار متطلب جديد
new_requirement = "The mobile application shall have a sleek and intuitive design."

# الحصول على النتيجة
result = classifier(new_requirement)

# طباعة النتيجة
label_id = int(result[0]['label'].replace('LABEL_', ''))
label_map = {0: "Functional", 1: "Non-Functional"}
label = label_map.get(label_id, "Unknown")
score = result[0]['score']

print(f"==================================")
print(f"المتطلب: {new_requirement}")
print(f"التصنيف: {label}, درجة الثقة: {score:.2%}")
print(f"==================================")