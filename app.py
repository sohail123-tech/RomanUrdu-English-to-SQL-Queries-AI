
import re
import requests
import pyodbc
import sqlglot
from sqlglot import exp

MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Change SERVER if your SQL Server instance has a different name.
CONNECTION_STRING = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    r"SERVER=localhost\SQLEXPRESS;"
    "DATABASE=RomanUrduSQLTest;"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

SCHEMA = """
Database: RomanUrduSQLTest
Dialect: Microsoft SQL Server (T-SQL)

Table: dbo.Patients
- PatientID INT
- PatientName NVARCHAR(100)
- Age INT
- City NVARCHAR(100)
- Gender NVARCHAR(20)
"""

SYSTEM_PROMPT = """
You are a text-to-SQL assistant for Microsoft SQL Server.

Understand questions in English and Roman Urdu.
Use only the schema provided by the application.
Generate exactly one read-only SELECT statement.
Use SQL Server T-SQL syntax.
Use only dbo.Patients and its listed columns.
Never generate INSERT, UPDATE, DELETE, DROP, ALTER,
CREATE, EXEC, MERGE, TRUNCATE, or stored procedure calls.
Never invent table or column names.
Do not use SELECT INTO, system tables, or other tables.
Do not use multiple statements, comments, or dynamic SQL.
Use TOP (100) to limit results to 100 rows.
If the question is unclear or cannot be answered from
the schema, return exactly: CANNOT_ANSWER

Return only SQL or CANNOT_ANSWER.
"""


def generate_sql(question):
    prompt = (
        SYSTEM_PROMPT
        + "\n\nSCHEMA:\n" + SCHEMA
        + "\n\nUSER QUESTION:\n" + question
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 200
            }
        },
        timeout=180
    )

    response.raise_for_status()
    data = response.json()
    sql = data["message"]["content"].strip()

    # Remove optional Markdown fences.
    sql = re.sub(
        r"\A```(?:sql)?[ \t]*\r?\n?",
        "",
        sql,
        flags=re.IGNORECASE
    )
    sql = re.sub(r"\r?\n?```[ \t]*\Z", "", sql).strip()

    return sql


def validate_sql(sql):
    if not sql:
        return False, "The model returned an empty response."

    if sql.strip().upper() == "CANNOT_ANSWER":
        return False, "The model cannot answer this question."

    # Reject SQL comments and multiple-statement separators.
    # Semicolons inside string literals are rejected too,
    # intentionally, to keep this prototype restrictive.
    if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
        return False, "Comments and semicolons are not allowed."

    try:
        statements = sqlglot.parse(sql, read="tsql")
    except Exception:
        return False, "SQL syntax could not be parsed."

    if len(statements) != 1 or statements[0] is None:
        return False, "Only one SQL statement is allowed."

    tree = statements[0]

    # Only a SELECT statement is permitted.
    if not isinstance(tree, exp.Select):
        return False, "Only SELECT statements are allowed."

    # Block SELECT INTO.
    if tree.args.get("into") is not None:
        return False, "SELECT INTO is not allowed."

    # Reject constructs that are unnecessary for this schema.
    forbidden_nodes = (
        exp.Command,
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Merge,
        exp.Union,
        exp.Intersect,
        exp.Except,
        exp.CTE,
        exp.Subquery,
    )

    for node in tree.walk():
        if isinstance(node, forbidden_nodes):
            return False, (
                f"Unsupported SQL construct: "
                f"{type(node).__name__}"
            )

    # Restrict every referenced table to dbo.Patients.
    for table in tree.find_all(exp.Table):
        if table.name.lower() != "patients":
            return False, f"Unauthorized table: {table.name}"

        if table.db and table.db.lower() != "dbo":
            return False, "Only the dbo schema is allowed."

        # Block three-part and four-part names.
        if table.catalog:
            return False, "Cross-database references are not allowed."

    # Restrict referenced columns to the supplied schema.
    allowed_columns = {
        "patientid",
        "patientname",
        "age",
        "city",
        "gender",
    }

    for column in tree.find_all(exp.Column):
        if column.is_star:
            continue

        if column.name.lower() not in allowed_columns:
            return False, f"Unauthorized column: {column.name}"

        # A qualified column must use the approved table name
        # or a table alias.
        if column.table:
            valid_qualifiers = {"patients"}
            valid_qualifiers.update(
                table.alias.lower()
                for table in tree.find_all(exp.Table)
                if table.alias
            )

            if column.table.lower() not in valid_qualifiers:
                return False, "Unauthorized table qualifier."

    # Require TOP to limit the output.
    if tree.args.get("top") is None:
        return False, "Query must include TOP (100) or a smaller TOP."

    return True, "Basic validation passed."


def execute_sql(sql):
    # IMPORTANT:
    # The SQL Server login should have SELECT permission only
    # on dbo.Patients, not general database permissions.

    with pyodbc.connect(
        CONNECTION_STRING,
        timeout=5,
        autocommit=True
    ) as connection:
        cursor = connection.cursor()

        # Limit query execution time and returned row count.
        cursor.execute("SET LOCK_TIMEOUT 5000;")
        cursor.execute("SET ROWCOUNT 100;")

        try:
            cursor.execute(sql)

            if cursor.description is None:
                raise RuntimeError("No result set was returned.")

            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchmany(100)

            return columns, rows

        finally:
            cursor.execute("SET ROWCOUNT 0;")


def main():
    print("Roman Urdu / English to SQL")
    print("Type exit to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "exit":
            print("Program closed.")
            break

        if not question:
            continue

        try:
            sql = generate_sql(question)

            print("\nGenerated SQL:\n", sql)

            safe, reason = validate_sql(sql)
            print("\nValidation:", reason)

            if not safe:
                continue

            confirm = input(
                "Execute on test database? (yes/no): "
            ).strip().lower()

            if confirm != "yes":
                print("Execution cancelled.\n")
                continue

            columns, rows = execute_sql(sql)

            print("\nColumns:", columns)

            for row in rows:
                print(tuple(row))

            print(f"\nRows returned: {len(rows)}\n")

        except requests.exceptions.ConnectionError:
            print(
                "Error: Cannot connect to Ollama. "
                "Make sure Ollama is installed and running.\n"
            )

        except requests.exceptions.Timeout:
            print("Error: Ollama request timed out.\n")

        except requests.exceptions.RequestException as error:
            print("Ollama API error:", error, "\n")

        except pyodbc.Error as error:
            print("SQL Server connection or query error:", error, "\n")

        except Exception as error:
            print("Error:", error, "\n")


if __name__ == "__main__":
    main()
