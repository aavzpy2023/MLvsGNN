# Build the manuscript in build/ so the auxiliary files stay out of the repository root.
# The PDF is copied back to the root after every successful build, because paper_ML_GNN.pdf is versioned there.
$pdf_mode = 1;
$out_dir = 'build';
$success_cmd = 'cp %Z%R.pdf %R.pdf';
