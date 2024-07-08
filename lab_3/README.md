### Етапи SVD

1. **Побудова матриці**: Створення матриці \( A \), де рядки відповідають користувачам, а стовпці — об'єктам.

2. **Розклад матриці**: Виконання сингулярного розкладу матриці \( A = U \Sigma V^T \):
   - \( U \): ортогональна матриця, що містить ліві сингулярні вектори.
   - \( \Sigma \): діагональна матриця зі сингулярними числами на діагоналі.
   - \( V \): ортогональна матриця, що містить праві сингулярні вектори.

3. **Зменшення розмірності**: Залишення лише перших \( k \) найбільших сингулярних чисел у \( \Sigma \) та відповідних їм стовпців у \( U \) та \( V \).

4. **Реконструкція матриці**: Повернення до вихідної матриці за допомогою зменшених розмірностей.

### Сфери застосування SVD

- Підбір рекомендацій для користувачів на основі їхньої історії взаємодії з системою.
- Зменшення кількості змінних у даних з метою зменшення обчислювальної складності.
- Обробка зображень для зменшення шуму.
- Аналіз тексту.
- Виявлення основних компонент у великих наборах даних.

### Вплив параметра \( k \)

- Велике \( k \): Зберігає більше сингулярних чисел, що призводить до більш точної реконструкції матриці, але може зберігати більше шуму та ускладнювати модель.
- Мале \( k \): Призводить до спрощення моделі, видалення шуму, але може втратити важливу інформацію, що впливає на точність передбачень. Вибір \( k \) залежить від балансу між точністю та складністю моделі.

### Переваги та недоліки SVD

**Переваги:**
- Зменшення кількості змінних, зберігаючи основну інформацію.
- Видалення шуму з даних при виборі менших значень \( k \).
- Простота реалізації.

**Недоліки:**
- Обчислення SVD для великих матриць може бути ресурсомістким.
- Складність інтерпретації сингулярних векторів.
- Поганий виконання у випадку, якщо матриця має багато пропущених значень.