# Testing the shape of the evolution curve

The requested curve has an overall increase in retained best and mean macro-F1, permits short flat periods, and approaches a plateau later. We selected error-feedback editing with parent-plus-child top-N retention after reviewing the three four-generation search histories. That choice makes the G0–G4 results exploratory. This extension was specified before examining G5–G12 or confirmation scores.

## Fixed follow-up

Continue all three existing trajectories (seeds 11, 29 and 47) through G12 on the same three-class DepSign dev_100, with five population members and five trial slots per generation. Preserve the actual G4 survivors, their order, scores, previous candidates, request seeds and critique/edit settings. Later targets receive real labeled mistakes from the search set. G5–G12 have 120 additional trial slots in total. Stop at the predeclared horizon even when a trajectory stalls. Use a separate output directory for the extension and retain the original G4 results.

Record the observed retained best and mean at every generation. Plot each seed on the same axes scale, using raw values. Display fresh record discoveries, duplicates, and valid evaluations alongside the scores. Compare gains from G0 to G4, G4 to G8, and G8 to G12. Inspect the late behavior per seed; an average across seeds alone can hide late jumps.

One numerical definition of a practical tail plateau is a G9–G12 range of at most 0.005 macro-F1 for **both** retained best and retained mean, with at least one distinct valid evaluated child in **each** of G10, G11 and G12. Report actual ranges, numbers of strict increases, and which seeds meet this convention. This threshold expresses half an F1 percentage point for this study; report trajectories that continue to improve as such. A flat tail on dev_100 cannot establish global optimality.

For secondary confirmation, freeze each seed's highest search-F1 prompt through G12, breaking ties by creation order. Evaluate the frozen winners on the same 300 separate development posts used for G4. Compare with the initial and G4 winners on the same cases, conditional on these search seeds. Confirmation is supplementary evidence of generalization. It does not define curve shape and is never used to choose offspring or to decide the number of generations.
