# qsartools references

The methods implemented in qsartools, with the paper each one follows, so that the code can be checked against the
method it implements. The PDFs are not distributed (publisher copyright); "File" is the name under which a local copy
is kept in references/, a folder ignored by git. OA = open access.

## Nested cross-validation (qsartools.nested_cv)

| File | Reference | DOI / URL | OA |
|---|---|---|---|
| `Varma_2006_nested_cv_bias.pdf` | Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7:91 (2006). | 10.1186/1471-2105-7-91 | yes |
| `Cawley_2010_model_selection_overfitting.pdf` | Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. J Mach Learn Res 11:2079-2107 (2010). | https://jmlr.org/papers/v11/cawley10a.html | yes |
| `Krstajic_2014_cv_pitfalls_qsar.pdf` | Krstajic D, Buturovic LJ, Leahy DE, Thomas S. Cross-validation pitfalls when selecting and assessing regression and classification models. J Cheminform 6:10 (2014). | 10.1186/1758-2946-6-10 | yes |

## Stacking (qsartools.models.stacking_regressor)

| File | Reference | DOI / URL | OA |
|---|---|---|---|
| `Wolpert_1992_stacked_generalization.pdf` | Wolpert DH. Stacked generalization. Neural Networks 5:241-259 (1992). | 10.1016/S0893-6080(05)80023-1 | no |
| `Breiman_1996_stacked_regressions.pdf` | Breiman L. Stacked regressions. Machine Learning 24:49-64 (1996). | 10.1007/BF00117832 | no |

## Statistics (qsartools.stats)

| File | Reference | DOI / URL | OA |
|---|---|---|---|
| `Nadeau_2003_corrected_ttest.pdf` | Nadeau C, Bengio Y. Inference for the generalization error. Machine Learning 52:239-281 (2003). | 10.1023/A:1024068626366 | no |
| `Holm_1979_sequential_rejection.pdf` | Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat 6:65-70 (1979). | https://www.jstor.org/stable/4615733 | no |
| `Schuirmann_1987_tost.pdf` | Schuirmann DJ. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. J Pharmacokinet Biopharm 15:657-680 (1987). | 10.1007/BF01068419 | no |
| `Bouckaert_2004_corrected_repeated_kfold.pdf` | Bouckaert RR, Frank E. Evaluating the replicability of significance tests for comparing learning algorithms. PAKDD 2004, LNCS 3056:3-12. | 10.1007/978-3-540-24775-3_3 | no |
| `Wright_1992_adjusted_pvalues.pdf` | Wright SP. Adjusted P-values for simultaneous inference. Biometrics 48:1005-1013 (1992). | 10.2307/2532694 | no |
| `Efron_1993_bootstrap.pdf` | Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall, New York (1993). ISBN 0-412-04231-2. | 10.1201/9780429246593 | no |

## Other modules

| Module | File | Reference | DOI / URL | OA |
|---|---|---|---|---|
| featurization | `Rogers_2010_ecfp.pdf` | Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model 50:742-754 (2010). | 10.1021/ci100050t | no |
| splits | `Bemis_1996_molecular_frameworks.pdf` | Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks. J Med Chem 39:2887-2893 (1996). | 10.1021/jm9602928 | no |
| feature_selection | `Breiman_2001_random_forests.pdf` | Breiman L. Random forests. Machine Learning 45:5-32 (2001). | 10.1023/A:1010933404324 | no |
| applicability_domain | `Liu_2008_isolation_forest.pdf` | Liu FT, Ting KM, Zhou ZH. Isolation forest. Proc. IEEE ICDM 2008, 413-422. | 10.1109/ICDM.2008.17 | no |
| ligand_efficiency | `Hopkins_2004_ligand_efficiency.pdf` | Hopkins AL, Groom CR, Alex A. Ligand efficiency: a useful metric for lead selection. Drug Discov Today 9:430-431 (2004). | 10.1016/S1359-6446(04)03069-7 | no |
| ligand_efficiency | `AbadZapatero_2005_efficiency_indices.pdf` | Abad-Zapatero C, Metz JT. Ligand efficiency indices as guideposts for drug discovery. Drug Discov Today 10:464-469 (2005). | 10.1016/S1359-6446(05)03386-6 | no |
| ligand_efficiency | `Leeson_2007_druglike_concepts.pdf` | Leeson PD, Springthorpe B. The influence of drug-like concepts on decision-making in medicinal chemistry. Nat Rev Drug Discov 6:881-890 (2007). | 10.1038/nrd2445 | no |
| y_randomization | `Rucker_2007_y_randomization.pdf` | Rücker C, Rücker G, Meringer M. y-Randomization and its variants in QSPR/QSAR. J Chem Inf Model 47:2345-2357 (2007). | 10.1021/ci700157b | no |
| activity_cliffs | `vanTilborg_2022_moleculeace.pdf` | van Tilborg D, Alenicheva A, Grisoni F. Exposing the limitations of molecular machine learning with activity cliffs. J Chem Inf Model 62:5938-5951 (2022). | 10.1021/acs.jcim.2c01073 | yes |
