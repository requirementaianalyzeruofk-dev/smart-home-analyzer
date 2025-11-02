import sqlite3

# استبدلي 'requirements (1).db' باسم الملف الخاص بك إذا لزم الأمر
db_name = 'requirements (1).db'

# حاول الاتصال بقاعدة البيانات
try:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # استعلام للحصول على أسماء الجداول
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    # استرجاع وإظهار أسماء الجداول
    tables = cursor.fetchall()
    print("الجداول الموجودة في قاعدة البيانات:")
    for table in tables:
        print(table[0])  # الجدول يظهر ك tuple، لذا نأخذ العنصر الأول

except sqlite3.Error as e:
    print("خطأ في الاتصال بقاعدة البيانات:", e)

finally:
    if conn:
        conn.close()  # تأكد من إغلاق الاتصال بقاعدة البيانات