# Naming Conventions Quick Notes

## 🐍 snake_case
- Words are separated by underscores.
- All lowercase.
- Common in Python, Rust, Postgres, Linux systems.

**Examples:**  
`user_name`, `total_score`, `created_at`

**When to use:**
- Variables  
- Functions  
- Database column names  
- File names  

---

## 🐪 camelCase
- First word lowercase, following words start with uppercase.
- Common in JavaScript, Java, Go.

**Examples:**  
`userName`, `totalScore`, `createdAt`

**When to use:**
- JS variables  
- JS functions  
- Client-side API objects  

---

## 🐫 PascalCase (UpperCamelCase)
- Each word starts with an uppercase letter.
- Preferred for type names.

**Examples:**  
`User`, `EventUpdate`, `MongoDateTime`

**When to use:**
- Struct / Class / Enum names  
- UI component names (React, Leptos)  
- Type definitions  

---

## 🍢 kebab-case
- Words separated by hyphens (`-`).
- Common for URLs and file names.

**Examples:**  
`user-profile`, `medical-record`, `event-update`

**When to use:**
- URL slugs  
- Web component file names  
- Package / config identifiers  

---

## 📐 SCREAMING_SNAKE_CASE
- All uppercase + underscores.
- Used for constants.

**Examples:**  
`MAX_SIZE`, `DEFAULT_LANG`, `TIMEOUT_MS`

**When to use:**
- Global constants  
- Config constants  
- Enum values (in some languages)  

---

# ⭐ Quick Recommendations
- **Rust:** `snake_case` for variables/functions, `PascalCase` for structs/enums, `SCREAMING_SNAKE_CASE` for constants.  
- **JS/TS:** `camelCase` for variables, `PascalCase` for components/classes, `kebab-case` for files/URLs.  
- **Databases:** Prefer `snake_case` for columns.  
- **URL slugs:** Use `kebab-case`.  
- **Typst / Markdown:** File names should be `kebab-case`.
