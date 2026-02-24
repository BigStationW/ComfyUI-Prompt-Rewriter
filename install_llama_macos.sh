#!/bin/bash

# ==========================================
#  llama.cpp Setup Script for macOS
#  Installs llama.cpp via Homebrew and creates
#  symlink structure for ComfyUI Prompt Rewriter
# ==========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[llama.cpp Setup]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BINARIES_DIR="${SCRIPT_DIR}/llama_binaries_macos"

log "Starting llama.cpp setup for macOS..."
log "Script directory: ${SCRIPT_DIR}"
log "Target symlink directory: ${LLAMA_BINARIES_DIR}"

# ==========================================
#  Check if Homebrew is installed
# ==========================================
if ! command -v brew &> /dev/null; then
    log_error "Homebrew is not installed. Please install Homebrew first:"
    log_error "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

log_success "Homebrew is installed"

# ==========================================
#  Check if llama.cpp is already installed
# ==========================================
if brew list llama.cpp &> /dev/null 2>&1; then
    log_success "llama.cpp is already installed via Homebrew"
    
    # Get current version
    LLAMA_VERSION=$(brew info llama.cpp | head -1 | awk '{print $3}' | sed 's/,$//')
    log "Current version: ${LLAMA_VERSION}"
    
    # Ask if user wants to update
    echo -n "${YELLOW}Do you want to update llama.cpp to the latest version? [y/N]:${NC} "
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log "Updating llama.cpp..."
        brew upgrade llama.cpp
        log_success "llama.cpp updated"
    else
        log "Keeping current version"
    fi
else
    log "llama.cpp is not installed. Installing via Homebrew..."
    brew install llama.cpp
    log_success "llama.cpp installed successfully"
fi

# ==========================================
#  Get llama.cpp installation path
# ==========================================
LLAMA_BREW_PREFIX=$(brew --prefix llama.cpp)
LLAMA_BIN_PATH="${LLAMA_BREW_PREFIX}/bin/llama-server"

if [[ ! -f "$LLAMA_BIN_PATH" ]]; then
    log_error "llama-server not found at ${LLAMA_BIN_PATH}"
    log_error "Please check your Homebrew installation"
    exit 1
fi

log_success "Found llama-server at: ${LLAMA_BIN_PATH}"

# ==========================================
#  Remove existing llama_binaries_macos directory if it exists
# ==========================================
if [[ -d "$LLAMA_BINARIES_DIR" ]]; then
    log_warning "Removing existing llama_binaries_macos directory..."
    rm -rf "$LLAMA_BINARIES_DIR"
fi

# ==========================================
#  Create symlink structure
# ==========================================
log "Creating symlink structure..."

# Create the target directory
mkdir -p "$LLAMA_BINARIES_DIR"

# Create symlinks to all relevant binaries and files
cd "$LLAMA_BINARIES_DIR"

# Find all binaries and create symlinks
find "$LLAMA_BREW_PREFIX/bin" -name "llama*" -type f -executable | while read -r binary; do
    binary_name=$(basename "$binary")
    log "Creating symlink: ${binary_name} -> ${binary}"
    ln -sf "$binary" "$binary_name"
done

# Also symlink any necessary libraries or share files if they exist
if [[ -d "$LLAMA_BREW_PREFIX/share" ]]; then
    mkdir -p share
    find "$LLAMA_BREW_PREFIX/share" -name "*llama*" -type d | while read -r share_dir; do
        dir_name=$(basename "$share_dir")
        log "Creating symlink for share directory: ${dir_name}"
        ln -sf "$share_dir" "share/${dir_name}"
    done
fi

# ==========================================
#  Verify installation
# ==========================================
log "Verifying installation..."

if [[ -f "${LLAMA_BINARIES_DIR}/llama-server" ]]; then
    log_success "llama-server symlink created successfully"
    
    # Test if the executable works
    if "${LLAMA_BINARIES_DIR}/llama-server" --help &> /dev/null; then
        log_success "llama-server is executable and responding"
    else
        log_warning "llama-server symlink created but executable test failed"
    fi
else
    log_error "Failed to create llama-server symlink"
    exit 1
fi

# ==========================================
#  Check Metal support
# ==========================================
log "Checking Metal support..."

if "${LLAMA_BINARIES_DIR}/llama-server" --help 2>&1 | grep -i metal &> /dev/null; then
    log_success "Metal backend support detected"
else
    log_warning "Metal backend support not explicitly detected (this may be normal)"
fi

# ==========================================
#  Final summary
# ==========================================
echo
log_success "========================================"
log_success "  llama.cpp setup completed successfully!"
log_success "========================================"
log_success "Symlink directory: ${LLAMA_BINARIES_DIR}"
log_success "llama-server: ${LLAMA_BINARIES_DIR}/llama-server"
log_success "Source installation: ${LLAMA_BREW_PREFIX}"
echo
log "You can now use the 'Metal' backend in ComfyUI Prompt Rewriter"
log "The llama-server executable will be automatically detected"
echo

# Show directory contents
log "Created symlinks:"
ls -la "$LLAMA_BINARIES_DIR" | head -10

echo
log_success "Setup complete!"
