# Değişken / Fonksiyon / Dosya Adlandırma Mini Notu

## 🐍 snake_case
- Kelimeler alt çizgi ile ayrılır.
- Tamamı küçük harf olur.
- Python, Rust, Postgres, Linux dünyasında yaygın.

**Örnekler:**  
`user_name`, `total_score`, `created_at`

**Ne zaman kullanılır?**
- Değişkenler
- Fonksiyonlar
- DB kolonları
- Dosya adları

---

## 🐪 camelCase
- İlk kelime küçük, diğer kelimeler büyük harfle başlar.
- JavaScript, Java, Go gibi dillerde yaygın.

**Örnekler:**  
`userName`, `totalScore`, `createdAt`

**Ne zaman kullanılır?**
- JS değişkenleri
- JS fonksiyonları
- API client tarafı

---

## 🐫 PascalCase (UpperCamelCase)
- Her kelime büyük harfle başlar.
- Class, Enum, Struct gibi yapılarda tercih edilir.

**Örnekler:**  
`User`, `EventUpdate`, `MongoDateTime`

**Ne zaman kullanılır?**
- Struct / Class / Enum isimleri
- Component adları (React, Leptos)
- Type isimleri

---

## 🍢 kebab-case
- Kelimeler tire (-) ile ayrılır.
- URL ve dosya adlarında yaygın.

**Örnekler:**  
`user-profile`, `medical-record`, `event-update`

**Ne zaman kullanılır?**
- URL slug
- Web bileşeni dosya adları
- Paket / config isimleri

---

## 📐 SCREAMING_SNAKE_CASE
- Tamamı büyük harf + alt çizgi.
- Sabitler için kullanılır.

**Örnekler:**  
`MAX_SIZE`, `DEFAULT_LANG`, `TIMEOUT_MS`

**Ne zaman kullanılır?**
- Global sabitler
- Config sabitleri
- Enum değerleri (bazı dillerde)

---

# ⭐ Kısa Öneriler
- **Rust:** `snake_case` (fonksiyon, değişken), `PascalCase` (struct/enum), `SCREAMING_SNAKE_CASE` (sabit).
- **JS/TS:** `camelCase` (değişken), `PascalCase` (component/class), `kebab-case` (dosyalar, URL).
- **Database:** Kolon isimleri için `snake_case`.
- **URL slug:** `kebab-case`.
- **Typst / Markdown:** Dosya adları `kebab-case`.
