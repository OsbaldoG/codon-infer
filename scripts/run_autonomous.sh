#!/usr/bin/env bash
set -u

BRANCH_NAME="feat/autonomous-nightly-$(date +%Y%m%d)"
git checkout -b "$BRANCH_NAME" 2>/dev/null || git checkout "$BRANCH_NAME"

TODO_FILE="tasks.todo"
LOG_FILE="autonomous_session.log"

echo "=== Iniciando Sesión Autónoma: $(date) ===" | tee -a "$LOG_FILE"

while IFS= read -r line || [ -n "$line" ]; do
    # Omitir líneas vacías o completadas
    [[ -z "$line" || "$line" =~ ^#.* || "$line" =~ ^DONE: ]] && continue

    echo -e "\n[$(date +%T)] Ejecutando: $line" | tee -a "$LOG_FILE"

    # Ejecutar OpenCode en modo no interactivo
    PROMPT="$line. Recuerda seguir las reglas de AGENTS.md, verificar con 'cargo test' y actualizar CHECKPOINTS.md."
    opencode run "$PROMPT" 2>&1 | tee -a "$LOG_FILE"

    # Verificación de integridad mediante el compilador
    echo "[$(date +%T)] Verificando compilación y pruebas..." | tee -a "$LOG_FILE"
    if cargo test --quiet; then
        echo "[$(date +%T)] Checkpoint superado. Creando commit..." | tee -a "$LOG_FILE"
        git add -A
        git commit -m "auto($BRANCH_NAME): ${line:0:50}" || true
        # Marcar tarea como completada en tasks.todo
        sed -i "s|^$line|DONE: $line|" "$TODO_FILE"
    else
        echo "[$(date +%T)] ERROR: Las pruebas fallaron tras la tarea. Revirtiendo cambios destructivos..." | tee -a "$LOG_FILE"
        git checkout -- src/ Cargo.toml
        echo "FAILED: $line" >> "$LOG_FILE"
        break
    fi

    # Pausa de cortesía para evitar saturación de tasa de API
    sleep 10
done < "$TODO_FILE"

echo "=== Sesión finalizada: $(date) ===" | tee -a "$LOG_FILE"