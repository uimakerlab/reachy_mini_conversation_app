#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Compile friction results markdown files to PDF.
#
# USAGE:
#   ./scripts/build_pdf.sh              # build blog.pdf
#   ./scripts/build_pdf.sh --all        # build blog.pdf + FRICTION_RESULTS.pdf
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)/friction_results"

build_pdf() {
    local src="$1"
    local name
    name="$(basename "${src}" .md)"
    local out="${DIR}/${name}.pdf"

    echo "==> Building ${out}..."
    pandoc "${src}" \
        --from=markdown \
        --to=pdf \
        --pdf-engine=xelatex \
        --resource-path="${DIR}" \
        --variable geometry:margin=2.5cm \
        --variable fontsize=11pt \
        --variable colorlinks=true \
        --variable linkcolor=blue \
        --variable urlcolor=blue \
        --highlight-style=tango \
        -o "${out}"
    echo "    -> ${out} ($(du -h "${out}" | cut -f1))"
}

# Try xelatex first, fall back to weasyprint, then wkhtmltopdf
if ! command -v xelatex &>/dev/null; then
    echo "xelatex not found, trying alternative PDF engines..."
    if command -v weasyprint &>/dev/null; then
        PDF_ENGINE="--pdf-engine=weasyprint"
    elif command -v wkhtmltopdf &>/dev/null; then
        PDF_ENGINE="--pdf-engine=wkhtmltopdf"
    else
        echo "No PDF engine found. Install texlive-xetex, weasyprint, or wkhtmltopdf."
        echo "  sudo apt install texlive-xetex texlive-fonts-recommended"
        echo "  # or: pip install weasyprint"
        exit 1
    fi
    # Override build_pdf to use detected engine
    build_pdf() {
        local src="$1"
        local name
        name="$(basename "${src}" .md)"
        local out="${DIR}/${name}.pdf"
        echo "==> Building ${out}..."
        pandoc "${src}" \
            --from=markdown \
            --to=pdf \
            ${PDF_ENGINE} \
            --resource-path="${DIR}" \
            -o "${out}"
        echo "    -> ${out} ($(du -h "${out}" | cut -f1))"
    }
fi

build_pdf "${DIR}/blog.md"

if [[ "${1:-}" == "--all" ]]; then
    build_pdf "${DIR}/FRICTION_RESULTS.md"
    build_pdf "${DIR}/FRICTION_ANALYSIS.md"
fi

echo ""
echo "Done. PDFs in ${DIR}/"
