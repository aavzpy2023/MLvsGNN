#!/usr/bin/env bash
# Build one of the manuscript layouts, or all of them. They share sections/ and latex/ and differ only in the class,
# the author block and the bibliography style, so adding a journal means adding one root .tex and nothing else.
#
#   ./build.sh                    every document, asking first which checks to run
#   ./build.sh paper_reading      just that one, no checks
#   ./build.sh --prose            the manuscript checks (6 s), then every document
#   ./build.sh --code             the whole test suite, then every document
#   ./build.sh --step             refresh the macro catalogue, the whole suite, then every document
#   ./build.sh --no-checks        straight to compiling (what a hook or another script should use)
#
# The question exists because the checks that matter depend on what changed, and running all of them every time costs
# forty seconds instead of six. It is asked only when the script is attached to a terminal and no document and no
# --flag were given, so every scripted use keeps compiling without a prompt. Committing and pushing are not this
# script's business.
#
# Everything is driven through latexmk so that .latexmkrc applies: the auxiliary files stay in build/ and only the
# PDF is copied back to the root, where it is versioned. Calling pdflatex directly bypasses that and litters the root.
#
# With no argument the documents are discovered the same way scripts/11_check_manuscript.py discovers them: every
# *.tex in the root that declares a \documentclass. A layout added for the next journal is therefore built as soon
# as it exists, with nothing to register here.
#
# The documents at the time of writing:
#   paper_reading   plain article, figures and tables next to the text they belong to. For reading and review.
#   paper_ML_GNN    Elsevier cas-sc. Its class collects the figures at the end, which is the submission layout.
#   paper_springer  Springer Nature sn-jnl, for Journal of Cheminformatics.
#   supplementary_ML_GNN
cd "$(dirname "$0")" || exit 1
# The search paths for the Springer class and its styles are set in .latexmkrc, so the editor finds them too.

# The tests need RDKit, which the base interpreter does not have: prefer the project's environment, as the README
# documents, and let MLGNN_PY override it.
PY=${MLGNN_PY:-}
if [ -z "$PY" ]; then
  for candidate in "$HOME/anaconda3/envs/mlvsgnn_repro/bin/python" "$HOME/miniconda3/envs/mlvsgnn_repro/bin/python"; do
    [ -x "$candidate" ] && PY=$candidate && break
  done
  PY=${PY:-python}
fi
# The checks that only prose can break: hand-written numbers, stale names, a catalogue whose citation column moved
MANUSCRIPT_TESTS="tests/test_manuscript_constants.py tests/test_macro_invariants.py tests/test_no_new_literals.py
                  tests/test_named_entities.py tests/test_macro_catalogue.py tests/test_manuscript_layout.py
                  tests/test_latex_tables.py"

run_checks() {
  case $1 in
    prose)
      echo "[checks] manuscript only"
      $PY -m pytest $MANUSCRIPT_TESTS -q || return 1
      ;;
    code)
      echo "[checks] whole test suite"
      $PY -m pytest tests/ -q || return 1
      ;;
    step)
      echo "[checks] refreshing the macro catalogue, then the whole test suite"
      $PY run_pipeline.py --only 11 > /dev/null || return 1
      $PY -m pytest tests/ -q || return 1
      ;;
  esac
}

CHECKS=""
case ${1:-} in
  --prose) CHECKS=prose; shift ;;
  --code) CHECKS=code; shift ;;
  --step) CHECKS=step; shift ;;
  --no-checks) shift ;;
  -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

if [ "$#" -gt 0 ]; then
  DOCS=$*
else
  DOCS=$(grep -l '^[^%]*\\documentclass' ./*.tex | sed 's|^\./||; s|\.tex$||' | sort)
  # Only when a person is watching: a hook, a cron or another script gets the old behaviour
  if [ -z "$CHECKS" ] && [ -t 0 ]; then
    echo "What changed since the last build?"
    echo "  1) prose only          - the manuscript checks, 6 s"
    echo "  2) code or settings    - the whole test suite, 40 s"
    echo "  3) I re-ran a step     - refresh the catalogue, then the whole suite"
    echo "  4) nothing, just build - no checks"
    printf "Choice [1-4, default 1]: "
    read -r choice
    case ${choice:-1} in
      1) CHECKS=prose ;;
      2) CHECKS=code ;;
      3) CHECKS=step ;;
      4) CHECKS="" ;;
      *) echo "Not a choice; running the manuscript checks."; CHECKS=prose ;;
    esac
  fi
fi

if [ -n "$CHECKS" ] && ! run_checks "$CHECKS"; then
  echo "[checks] failed; nothing was compiled. Fix the test above, or compile anyway with --no-checks."
  exit 1
fi
[ -n "$DOCS" ] || { echo "No root .tex (one with \\documentclass) found"; exit 1; }
STATUS=0
for doc in $DOCS; do
  # Inside the loop, not once before it: another process (an editor's build, a parallel run of this script) can
  # remove build/ between the start and this document's turn, and the redirection below would then fail.
  mkdir -p build
  if ! latexmk -pdf -interaction=nonstopmode "$doc.tex" > "build/${doc}_build.log" 2>&1; then
    printf "  %-22s LATEXMK FAILED, see build/%s_build.log\n" "$doc" "$doc"
    tail -5 "build/${doc}_build.log" | sed 's/^/      /'
    STATUS=1
    continue
  fi
  # .latexmkrc copies the PDF back only after a build it considered necessary, and latexmk removes a stale copy
  # from the root, so a document that was already up to date would lose its PDF. Copy unconditionally.
  [ -f "build/$doc.pdf" ] && cp -u "build/$doc.pdf" "$doc.pdf"
  log="build/${doc}.log"
  errors=$(grep -cE '^! ' "$log" 2>/dev/null); errors=${errors:-0}
  undefined=$(grep -cE 'Citation .* undefined|Reference .* undefined' "$log" 2>/dev/null); undefined=${undefined:-0}
  pages=$(pdfinfo "$doc.pdf" 2>/dev/null | awk '/^Pages/{print $2" pages"}')
  printf "  %-22s errors=%-3s undefined=%-3s %s\n" "$doc" "$errors" "$undefined" "$pages"
  if [ "$errors" != "0" ] || [ "$undefined" != "0" ]; then STATUS=1; fi
done
exit $STATUS
