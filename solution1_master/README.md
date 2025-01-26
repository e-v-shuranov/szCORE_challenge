# Automatic recognition of epileptic seizures in the EEG

Repository for algorithms to be evaluated with [SzCORE](https://github.com/esl-epfl/szcore/).

The code is developed to run automated seizure detection on BIDS / SzCORE standardized `EDF` files. The code is expected to output a HED-SCORE / SzCORE `TSV` annotation file.

The python code is then packaged in a docker container. The container contains two volumes (`/data`, `/output`), used respectively, to hold the input `EDF` and output `TSV.` Upon entry the container should run the algorithm. It expects the input filename as an environment variable `$INPUT` and the output filename as an environment variable `$OUTPUT`.
