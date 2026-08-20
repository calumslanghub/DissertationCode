# Read This
The repo contains all of the raw data and coding framework used in the process of writing my Masters dissertation.

The process was spilt across several notebooks with the key data being transferred as csv files into the new notebooks.
The accompanying flow chart explains the workflow through the notebooks.

All files were ran using python version 3.14.3

Unfortunately the BSS dataset & Output Area.shp are too large to share to upload to the repo.

BSS Dataset: https://usmart.ai/organisation/cyclingscotland/discovery/558cb4f5-d119-4b95-9347-ee130946d86f 

Output Area SHP: https://www.nrscotland.gov.uk/publications/2022-census-geography-products/ 

The workbook flow is as follows: GlasgowBikesSRM -> CensusFix_2 -> GlasgowExposureFinal -> OnPremisesVariable -> commuterk-means -> RegressionRun

GlasgoeBikesSRM: is Data Cleaning.

CensusFix_2: Creates the Census Variables.

GlasgowExposureFinal: Creates the routes and cycle infrastructure exposure variables.

OnPremisesVariable: Creates Licensed Premises Variable.

commuterk-means: Runs the k-means commuter classification.

RegressionRun: Runs the NB regression model and robustness checks.

