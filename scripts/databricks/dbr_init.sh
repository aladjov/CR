#!/bin/bash
#
# Databricks Cluster Initialization Script
#
# Supports three deployment modes:
#   1. WHEEL MODE (Production): Install from pre-built wheel in UC Volume
#   2. REPO MODE (Development): Install from Workspace Repo with editable install
#   3. HUGGINGFACE MODE: Download and install from private HuggingFace repo
#
# Environment Variables:
#   DBR_INSTALL_MODE      - "wheel", "repo", or "huggingface" (default: auto-detect)
#   DBR_WHEEL_PATH        - Path to wheel file or directory containing wheels
#                           e.g., /Volumes/catalog/schema/packages/customer_retention-1.0.0-py3-none-any.whl
#                           e.g., /Volumes/catalog/schema/packages/ (finds latest)
#   DBR_PROJECT_PATH      - Path to project root (for repo mode)
#   DBR_HF_REPO           - HuggingFace repo (e.g., "your-org/customer-retention")
#   DBR_HF_FILENAME       - Wheel filename in HF repo (e.g., "customer_retention-1.0.0-py3-none-any.whl")
#   HF_TOKEN              - HuggingFace token (for private repos)
#   DBR_CONSTRAINTS_PATH  - Path to constraints file (optional)
#   DBR_QUIET             - Set to "1" to suppress output
#
# Logs:
#   /tmp/dbr_init.log
#

set -e

LOG_FILE="/tmp/dbr_init.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() {
    echo "[${TIMESTAMP}] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[${TIMESTAMP}] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# Initialize log file
echo "=== DBR Init Script Started ===" > "$LOG_FILE"
echo "Timestamp: ${TIMESTAMP}" >> "$LOG_FILE"
echo "Hostname: $(hostname)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Install uv if not present
install_uv() {
    log "Checking for uv..."
    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
        log "uv is already installed: $UV_VERSION"
        return 0
    fi

    log "Installing uv..."
    if pip install uv --quiet >> "$LOG_FILE" 2>&1; then
        UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
        log "uv installed successfully: $UV_VERSION"
        return 0
    else
        log_error "Failed to install uv"
        return 1
    fi
}

# Detect installation mode
detect_mode() {
    if [ -n "$DBR_INSTALL_MODE" ]; then
        echo "$DBR_INSTALL_MODE"
        return
    fi

    # Auto-detect based on which env vars are set
    if [ -n "$DBR_HF_REPO" ]; then
        echo "huggingface"
    elif [ -n "$DBR_WHEEL_PATH" ]; then
        echo "wheel"
    elif [ -n "$DBR_PROJECT_PATH" ]; then
        echo "repo"
    else
        echo "none"
    fi
}

# Find wheel file (supports directory or direct path)
find_wheel() {
    local wheel_path="$1"

    # Direct wheel file
    if [[ "$wheel_path" == *.whl ]] && [ -f "$wheel_path" ]; then
        echo "$wheel_path"
        return 0
    fi

    # Directory - find latest customer_retention wheel
    if [ -d "$wheel_path" ]; then
        local latest=$(ls -t "${wheel_path}"/customer_retention-*.whl 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "$latest"
            return 0
        fi
    fi

    return 1
}

# Find constraints file
find_constraints() {
    # Use explicitly provided path if set
    if [ -n "$DBR_CONSTRAINTS_PATH" ] && [ -f "$DBR_CONSTRAINTS_PATH" ]; then
        echo "$DBR_CONSTRAINTS_PATH"
        return 0
    fi

    # For repo mode, check project directory
    if [ -n "$DBR_PROJECT_PATH" ]; then
        # Try pyproject.toml setting
        local constraints_rel=$(python3 -c "
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import sys
path = '${DBR_PROJECT_PATH}/pyproject.toml'
try:
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    print(data.get('tool', {}).get('databricks', {}).get('constraints-path', ''))
except:
    pass
" 2>/dev/null || echo "")

        if [ -n "$constraints_rel" ]; then
            local constraints_path="${DBR_PROJECT_PATH}/${constraints_rel}"
            if [ -f "$constraints_path" ]; then
                echo "$constraints_path"
                return 0
            fi
        fi

        # Auto-detect from constraints directory
        local constraints_dir="${DBR_PROJECT_PATH}/constraints"
        if [ -d "$constraints_dir" ]; then
            local latest=$(ls -t "${constraints_dir}"/dbr-*.txt 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                echo "$latest"
                return 0
            fi
        fi
    fi

    echo ""
}

# Install from wheel (Production mode)
install_wheel() {
    local wheel_file=$(find_wheel "$DBR_WHEEL_PATH")

    if [ -z "$wheel_file" ]; then
        log_error "No wheel found at: $DBR_WHEEL_PATH"
        return 1
    fi

    log "Installing from wheel: $wheel_file"

    local constraints=$(find_constraints)
    local uv_cmd="uv pip install --system"

    if [ -n "$constraints" ]; then
        log "Using constraints: $constraints"
        uv_cmd="$uv_cmd -c $constraints"
    fi

    uv_cmd="$uv_cmd $wheel_file"

    if [ "$DBR_QUIET" = "1" ]; then
        uv_cmd="$uv_cmd --quiet"
    fi

    log "Running: $uv_cmd"

    if eval "$uv_cmd" >> "$LOG_FILE" 2>&1; then
        log "Wheel installed successfully"
        return 0
    else
        log_error "Wheel installation failed"
        return 1
    fi
}

# Install from repo (Development mode)
install_repo() {
    if [ -z "$DBR_PROJECT_PATH" ]; then
        log_error "DBR_PROJECT_PATH not set for repo mode"
        return 1
    fi

    if [ ! -d "$DBR_PROJECT_PATH" ]; then
        log_error "Project path does not exist: $DBR_PROJECT_PATH"
        return 1
    fi

    local pyproject="${DBR_PROJECT_PATH}/pyproject.toml"
    if [ ! -f "$pyproject" ]; then
        log_error "pyproject.toml not found: $pyproject"
        return 1
    fi

    log "Installing from repo: $DBR_PROJECT_PATH"

    local constraints=$(find_constraints)
    local uv_cmd="uv pip install --system"

    if [ -n "$constraints" ]; then
        log "Using constraints: $constraints"
        uv_cmd="$uv_cmd -c $constraints"
    fi

    # Install package in editable mode + dependencies
    uv_cmd="$uv_cmd -e $DBR_PROJECT_PATH"

    if [ "$DBR_QUIET" = "1" ]; then
        uv_cmd="$uv_cmd --quiet"
    fi

    log "Running: $uv_cmd"

    if eval "$uv_cmd" >> "$LOG_FILE" 2>&1; then
        log "Repo installed successfully"
        return 0
    else
        log_error "Repo installation failed"
        return 1
    fi
}

# Install from HuggingFace (Private repo mode)
install_huggingface() {
    if [ -z "$DBR_HF_REPO" ]; then
        log_error "DBR_HF_REPO not set for huggingface mode"
        return 1
    fi

    # Determine wheel filename
    local wheel_filename="${DBR_HF_FILENAME:-customer_retention-1.0.0-py3-none-any.whl}"

    # Build HuggingFace URL
    local hf_url="https://huggingface.co/${DBR_HF_REPO}/resolve/main/${wheel_filename}"

    log "Installing from HuggingFace: $hf_url"

    local constraints=$(find_constraints)
    local uv_cmd="uv pip install --system"

    if [ -n "$constraints" ]; then
        log "Using constraints: $constraints"
        uv_cmd="$uv_cmd -c $constraints"
    fi

    # For private repos, uv/pip will use HF_TOKEN env var automatically
    # Alternatively, we can embed it in the URL
    if [ -n "$HF_TOKEN" ]; then
        # Use authenticated URL format
        hf_url="https://hf-user:${HF_TOKEN}@huggingface.co/${DBR_HF_REPO}/resolve/main/${wheel_filename}"
        log "Using authenticated HuggingFace URL"
    fi

    uv_cmd="$uv_cmd $hf_url"

    if [ "$DBR_QUIET" = "1" ]; then
        uv_cmd="$uv_cmd --quiet"
    fi

    log "Running: uv pip install --system ... (URL hidden for security)"

    if eval "$uv_cmd" >> "$LOG_FILE" 2>&1; then
        log "HuggingFace package installed successfully"
        return 0
    else
        log_error "HuggingFace installation failed"
        return 1
    fi
}

# Log installed packages
log_packages() {
    log "Verifying customer_retention installation..."
    python3 -c "import customer_retention; print(f'customer_retention version: {customer_retention.__version__}')" >> "$LOG_FILE" 2>&1 || log "Warning: could not import customer_retention"
}

# Main execution
main() {
    local mode=$(detect_mode)
    log "Installation mode: $mode"

    if [ "$mode" = "none" ]; then
        log_error "No installation mode detected"
        log "Set DBR_WHEEL_PATH for wheel mode or DBR_PROJECT_PATH for repo mode"
        exit 0
    fi

    if ! install_uv; then
        log_error "uv installation failed, cannot continue"
        exit 0
    fi

    case "$mode" in
        wheel)
            if ! install_wheel; then
                log_error "Wheel installation failed"
                exit 0
            fi
            ;;
        repo)
            if ! install_repo; then
                log_error "Repo installation failed"
                exit 0
            fi
            ;;
        huggingface)
            if ! install_huggingface; then
                log_error "HuggingFace installation failed"
                exit 0
            fi
            ;;
        *)
            log_error "Unknown mode: $mode"
            exit 0
            ;;
    esac

    log_packages

    log "=== DBR Init Script Completed Successfully ==="
}

# Run main, but don't fail cluster startup on errors
main || {
    log_error "Init script encountered an error, but cluster startup will continue"
    exit 0
}

exit 0
