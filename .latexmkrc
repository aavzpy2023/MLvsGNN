# Build the manuscript in build/ so the auxiliary files stay out of the repository root.
# The PDF is copied back to the root after every successful build, because paper_ML_GNN.pdf is versioned there.
$pdf_mode = 1;
$out_dir = 'build';
$success_cmd = 'cp %Z%R.pdf %R.pdf';

# The Springer class and its bibliography styles live in templates/springer/, so that vendor files stay out of the
# project root. Setting the search paths here rather than in the shell means every caller finds them: the editor,
# logs/build.sh and a bare `latexmk`.
$ENV{'TEXINPUTS'} = './/:./templates/springer//:' . ($ENV{'TEXINPUTS'} // '');
$ENV{'BSTINPUTS'} = './/:./templates/springer/bst//:' . ($ENV{'BSTINPUTS'} // '');
$ENV{'BIBINPUTS'} = './/:' . ($ENV{'BIBINPUTS'} // '');
