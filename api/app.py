from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import jwt
import uuid
from typing import Optional, List
import os
import logging
import bcrypt

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(root_path="/api-v")

# Configuración
SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # Cambia en producción
ALGORITHM = "HS256"

# Modelos
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

# Esquema de autenticación
security = HTTPBearer()

# Inicialización de la base de datos
def init_db():
    """Inicializa la base de datos SQLite con tablas para máquinas, empleados, sesiones y usuarios."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Tabla para máquinas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                machine TEXT NOT NULL,
                production_line TEXT NOT NULL,
                material TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                uptime REAL NOT NULL,
                defects INTEGER NOT NULL,
                vibration REAL NOT NULL,
                temperature REAL NOT NULL,
                defect_type TEXT NOT NULL,
                throughput REAL NOT NULL,
                inventory_level INTEGER NOT NULL
            )
        """)
        
        # Tabla para empleados (date antes que hire_date)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                shift TEXT NOT NULL,
                assigned_machine TEXT NOT NULL,
                production_line TEXT NOT NULL,
                date TEXT NOT NULL,  
                hire_date TEXT NOT NULL,
                termination_date TEXT,
                hours_worked REAL NOT NULL,
                training_status TEXT NOT NULL
            )
        """)
        
        # Tabla para tokens de sesión (expiry ahora es opcional)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expiry DATETIME
            )
        """)
        
        # Tabla para usuarios
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        """)
        
        # Insertar registros fijos para máquinas
        cursor.execute("SELECT COUNT(*) FROM machines")
        count = cursor.fetchone()[0]
        if count == 0:
            fixed_records = [
                ("2025-04-10", "ModelA", "Line3", "Steel", "BATCH101", 95.0, 2, 0.7, 75.0, "scratch", 90.0, 400),
                ("2025-04-10", "ModelB", "Line2", "Aluminum", "BATCH102", 97.5, 1, 0.6, 70.0, "dent", 88.0, 350),
                ("2025-04-11", "ModelC", "Line1", "Copper", "BATCH137", 87.0, 3, 0.8, 81.0, "crack", 85.0, 350),
                ("2025-04-10", "ModelD", "Line3", "Plastic", "BATCH104", 97.5, 4, 0.9, 78.0, "warp", 87.0, 300),
                ("2025-04-10", "ModelE", "Line2", "Brass", "BATCH105", 90.0, 2, 0.65, 72.0, "chip", 89.0, 320),
                ("2025-04-10", "ModelF", "Line1", "Titanium", "BATCH106", 95.0, 2, 0.75, 76.0, "scratch", 91.0, 380),
                ("2025-04-09", "ModelA", "Line1", "Aluminum", "BATCH107", 97.5, 1, 0.6, 74.0, "dent", 92.0, 410),
                ("2025-04-09", "ModelB", "Line2", "Aluminum", "BATCH108", 92.0, 3, 0.8, 79.0, "crack", 86.0, 340),
                ("2025-04-09", "ModelC", "Line1", "Copper", "BATCH109", 88.5, 4, 0.85, 82.0, "warp", 84.0, 360),
                ("2025-04-09", "ModelD", "Line3", "Plastic", "BATCH110", 90.0, 2, 0.7, 77.0, "chip", 88.0, 310),
                ("2025-04-09", "ModelE", "Line2", "Brass", "BATCH111", 99.0, 0, 0.5, 70.0, "none", 93.0, 330),
                ("2025-04-09", "ModelF", "Line1", "Titanium", "BATCH112", 85.5, 5, 0.9, 80.0, "scratch", 83.0, 370),
                ("2025-04-11", "ModelA", "Line1", "Steel", "BATCH113", 93.0, 1, 0.4, 73.0, "dent", 90.0, 390),
                ("2025-04-11", "ModelB", "Line2", "Aluminum", "BATCH114", 96.5, 2, 0.7, 71.0, "crack", 89.0, 360),
                ("2025-04-11", "ModelD", "Line3", "Plastic", "BATCH115", 89.0, 3, 0.8, 79.0, "warp", 86.0, 320),
            ]
            cursor.executemany("""
                INSERT INTO machines (
                    date, machine, production_line, material, batch_id, uptime, defects,
                    vibration, temperature, defect_type, throughput, inventory_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, fixed_records)
        
        # Insertar registros fijos para empleados (date antes que hire_date en la tupla)
        cursor.execute("SELECT COUNT(*) FROM employees")
        count = cursor.fetchone()[0]
        if count == 0:
            fixed_records = [
                ("001", "John Doe", "Operator", "Morning", "ModelA", "Line3", "2025-04-10", "2024-01-15", None, 8.0, "Completed"),
                ("002", "Jane Smith", "Supervisor", "Evening", "ModelB", "Line2", "2025-04-10", "2023-06-20", None, 7.5, "In Progress"),
                ("003", "Carlos Lopez", "Technician", "Night", "ModelC", "Line1", "2025-04-11", "2024-03-10", None, 8.5, "Completed"),
                ("004", "Maria Garcia", "Operator", "Morning", "ModelD", "Line3", "2025-04-10", "2024-02-01", None, 8.0, "Completed"),
                ("005", "Ahmed Khan", "Supervisor", "Evening", "ModelE", "Line2", "2025-04-10", "2023-09-15", None, 7.0, "In Progress"),
                ("006", "Sophie Chen", "Operator", "Morning", "ModelF", "Line1", "2025-04-10", "2024-05-05", None, 8.0, "Completed"),
                ("007", "David Brown", "Technician", "Night", "ModelA", "Line1", "2025-04-09", "2023-11-30", None, 8.5, "Completed"),
                ("008", "Emma Wilson", "Operator", "Morning", "ModelB", "Line2", "2025-04-09", "2024-04-12", None, 7.5, "In Progress"),
                ("009", "Lucas Martinez", "Supervisor", "Evening", "ModelC", "Line1", "2025-04-09", "2023-07-25", None, 8.0, "Completed"),
                ("010", "Olivia Lee", "Operator", "Morning", "ModelD", "Line3", "2025-04-11", "2024-01-20", None, 8.0, "Completed"),
            ]
            cursor.executemany("""
                INSERT INTO employees (
                    employee_id, name, role, shift, assigned_machine, production_line,
                    date, hire_date, termination_date, hours_worked, training_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, fixed_records)
        
        conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        conn.close()

@app.on_event("startup")
async def startup_event():
    """Ejecuta la inicialización de la base de datos al iniciar la aplicación."""
    init_db()

# Generar token JWT
def create_jwt_token(username: str) -> str:
    """Genera un token JWT para un usuario dado sin tiempo de expiración."""
    try:
        to_encode = {"sub": username, "jti": str(uuid.uuid4())}
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Almacenar token en la base de datos con expiry NULL
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (token, username, expiry) VALUES (?, ?, ?)",
                       (token, username, None))
        conn.commit()
        return token
    except Exception as e:
        logger.error(f"Failed to create JWT token: {str(e)}")
        raise
    finally:
        conn.close()

# Validar token
async def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Valida un token JWT proporcionado en el encabezado de autorización."""
    try:
        token = credentials.credentials
        # Validar solo con JWT, sin verificar expiración
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token: No username")
        return username
    except jwt.PyJWTError as e:
        logger.error(f"JWT validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid or malformed token")
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Endpoint de registro
@app.post("/register")
async def register(request: RegisterRequest):
    """Registra un nuevo usuario en la base de datos."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Verificar si el usuario ya existe
        cursor.execute("SELECT username FROM users WHERE username = ?", (request.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Hashear la contraseña
        hashed_password = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Insertar nuevo usuario
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                       (request.username, hashed_password))
        conn.commit()
        
        # Generar token para el nuevo usuario
        token = create_jwt_token(request.username)
        
        return {"access_token": token, "token_type": "bearer", "message": "User registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint de login
@app.post("/login")
async def login(request: LoginRequest):
    """Autentica a un usuario y genera un token JWT."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (request.username,))
        user = cursor.fetchone()
        
        if not user or not bcrypt.checkpw(request.password.encode('utf-8'), user[0].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = create_jwt_token(request.username)
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint para obtener todos los registros de máquinas (protegido)
@app.get("/machines/")
async def get_all_machines(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    username: str = Depends(validate_token)
):
    """Obtiene todos los registros de máquinas, opcionalmente filtrados por fechas."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        base_query = """
            SELECT id, date, machine, production_line, material, batch_id, uptime, defects,
                   vibration, temperature, defect_type, throughput, inventory_level
            FROM machines
        """
        
        conditions = []
        params = []
        
        if specific_date:
            conditions.append("date = ?")
            params.append(specific_date)
        else:
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
        
        query = base_query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        
        machines = []
        for row in cursor.fetchall():
            machines.append({
                "id": row[0],
                "date": row[1],
                "machine": row[2],
                "production_line": row[3],
                "material": row[4],
                "batch_id": row[5],
                "uptime": row[6],
                "defects": row[7],
                "vibration": row[8],
                "temperature": row[9],
                "defect_type": row[10],
                "throughput": row[11],
                "inventory_level": row[12]
            })
        return machines
    except Exception as e:
        logger.error(f"Failed to fetch machines: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint para obtener registros por máquina (protegido)
@app.get("/machines/{machine}")
async def get_machine_records(
    machine: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    username: str = Depends(validate_token)
):
    """Obtiene registros para una máquina específica, opcionalmente filtrados por fechas."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        query = """
            SELECT id, date, machine, production_line, material, batch_id, uptime, defects,
                   vibration, temperature, defect_type, throughput, inventory_level
            FROM machines 
            WHERE machine = ?
        """
        
        params = [machine]
        conditions = []
        
        if specific_date:
            conditions.append("date = ?")
            params.append(specific_date)
        else:
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        
        records = []
        for row in cursor.fetchall():
            records.append({
                "id": row[0],
                "date": row[1],
                "machine": row[2],
                "production_line": row[3],
                "material": row[4],
                "batch_id": row[5],
                "uptime": row[6],
                "defects": row[7],
                "vibration": row[8],
                "temperature": row[9],
                "defect_type": row[10],
                "throughput": row[11],
                "inventory_level": row[12]
            })
        
        if not records:
            raise HTTPException(status_code=404, detail="Máquina no encontrada")
        
        return records
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch machine records: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint para obtener todos los registros de empleados (protegido)
@app.get("/employees/")
async def get_all_employees(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    username: str = Depends(validate_token)
):
    """Obtiene todos los registros de empleados, opcionalmente filtrados por fechas."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        base_query = """
            SELECT id, employee_id, name, role, shift, assigned_machine, production_line,
                   date, hire_date, termination_date, hours_worked, training_status
            FROM employees
        """
        
        conditions = []
        params = []
        
        if specific_date:
            conditions.append("date = ?")
            params.append(specific_date)
        else:
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
        
        query = base_query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        
        employees = []
        for row in cursor.fetchall():
            employees.append({
                "id": row[0],
                "employee_id": row[1],
                "name": row[2],
                "role": row[3],
                "shift": row[4],
                "assigned_machine": row[5],
                "production_line": row[6],
                "date": row[7],  # Nuevo orden: date antes que hire_date
                "hire_date": row[8],
                "termination_date": row[9],
                "hours_worked": row[10],
                "training_status": row[11]
            })
        return employees
    except Exception as e:
        logger.error(f"Failed to fetch employees: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint para obtener registros por empleado (protegido)
@app.get("/employees/{employee_id}")
async def get_employee_records(
    employee_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    username: str = Depends(validate_token)
):
    """Obtiene registros para un empleado específico, opcionalmente filtrados por fechas."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        query = """
            SELECT id, employee_id, name, role, shift, assigned_machine, production_line,
                   date, hire_date, termination_date, hours_worked, training_status
            FROM employees 
            WHERE employee_id = ?
        """
        
        params = [employee_id]
        conditions = []
        
        if specific_date:
            conditions.append("date = ?")
            params.append(specific_date)
        else:
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        
        records = []
        for row in cursor.fetchall():
            records.append({
                "id": row[0],
                "employee_id": row[1],
                "name": row[2],
                "role": row[3],
                "shift": row[4],
                "assigned_machine": row[5],
                "production_line": row[6],
                "date": row[7],  # Nuevo orden: date antes que hire_date
                "hire_date": row[8],
                "termination_date": row[9],
                "hours_worked": row[10],
                "training_status": row[11]
            })
        
        if not records:
            raise HTTPException(status_code=404, detail="Empleado no encontrado")
        
        return records
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch employee records: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)