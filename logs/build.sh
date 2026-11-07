#!/usr/bin/env bash
# Build one of the manuscript layouts, or all of them. They share sections/ and latex/ and differ only in the class,
# the author block and the bibliography style, so adding a journal means adding one root .tex and nothing else.
#
#   logs/build.sh                 every document
#   logs/build.sh paper_reading   just that one
#
# Everything is driven through latexmk so that .latexmkrc applies: the auxiliary files stay in build/ and only the
# PDF is copied back to the root, where it is versioned. Calling pdflatex directly bypasses that and litters the root.
#
# Documents:
#   paper_reading   plain article, figures and tables next to the text they belong to. For reading and review.
#   paper_ML_GNN    Elsevier cas-sc. Its class collects the figures at the end, which is the submission layout.
#   paper_springer  Springer Nature sn-jnl, for Journal of Cheminformatics.
#   supplementary_ML_GNN
cd "$(dirname "$0")/.." || exit 1
export TEXINPUTS=".//:./templates/springer//:"
export BSTINPUTS=".//:./templates/springer/bst//:"

DOCS=${*:-"paper_reading paper_ML_GNN paper_springer supplementary_ML_GNN"}
STATUS=0
for doc in $DOCS; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error=false "$doc.tex" > "build/${doc}_build.log" 2>&1
  log="build/${doc}.log"
  errors=$(grep -cE '^! ' "$log" 2>/dev/null); errors=${errors:-0}
  undefined=$(grep -cE 'Citation .* undefined|Reference .* undefined' "$log" 2>/dev/null); undefined=${undefined:-0}
  pages=$(pdfinfo "$doc.pdf" 2>/dev/null | awk '/^Pages/{print $2" pages"}')
  printf "  %-22s errors=%-3s undefined=%-3s %s\n" "$doc" "$errors" "$undefined" "$pages"
  if [ "$errors" != "0" ] || [ "$undefined" != "0" ]; then STATUS=1; fi
done
exit $STATUS
