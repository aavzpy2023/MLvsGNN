#!/usr/bin/env bash
# Build one of the manuscript layouts, or all of them. They share sections/ and latex/ and differ only in the class,
# the author block and the bibliography style, so adding a journal means adding one root .tex and nothing else.
#
#   logs/build.sh                 every document
#   logs/build.sh paper_reading   just that one
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
  pdflatex -interaction=nonstopmode "$doc.tex" > "logs/${doc}_build.log" 2>&1
  bibtex "$doc" >> "logs/${doc}_build.log" 2>&1
  pdflatex -interaction=nonstopmode "$doc.tex" >> "logs/${doc}_build.log" 2>&1
  pdflatex -interaction=nonstopmode "$doc.tex" > "logs/${doc}_build.log" 2>&1
  errors=$(grep -cE '^! ' "logs/${doc}_build.log")
  undefined=$(grep -cE 'Citation .* undefined|Reference .* undefined' "logs/${doc}_build.log")
  pages=$(grep -oE '\([0-9]+ pages' "logs/${doc}_build.log" | head -1 | tr -d '(')
  printf "  %-22s errors=%-3s undefined=%-3s %s\n" "$doc" "$errors" "$undefined" "$pages"
  [ "$errors" != "0" ] || [ "$undefined" != "0" ] && STATUS=1
done
exit $STATUS
