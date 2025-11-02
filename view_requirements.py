from db_manager import get_requirements

# استرجاع المتطلبات من قاعدة البيانات
requirements = get_requirements()

# طباعة منظمة
print("\n قائمة المتطلبات:")
for req in requirements:
    print(f"- ID {req['id']}: {req['text']}")
    print(f"  Functional: {req['functional']}, Non-Functional: {req['non_functional']}")
    print(f"  Ambiguous: {req['ambiguous']}, Measurable: {req['measurable']}")
    print(f"  Notes: {req['notes']}")
    print(f"   File: {req['file']}")
    print("-" * 60)
