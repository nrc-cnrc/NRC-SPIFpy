# NRC SPIFpy

NRC's Single Particle Image Format (SPIF) conversion utility. 
Version 1.0 (Release date 9 December, 2021)

## About

**NRC-SPIFpy** is a set of Command Line Interface(CLI) tools which allow for the conversion of files stored in a 
variety of raw imaging probe formats to the **SPIF** format. The package is written in **Python**, 
and includes the following utilities:

- `nrc-spifpy-extract`: Convert a file in a raw imaging probe format to the **SPIF** format.
- `nrc-spifpy-addaux`: Add auxiliary data to a file in the **SPIF** format.
- `nrc-spifpy-cc`: Copy the configuration files required for processing with `spifpy` and `spifaddaux`.

## Installation Requirements
- Python 3.6+
- Linux/MacOS : Any python environment manager (pyenv preferred)
- Windows : [Anaconda Python Distribution(64-bit)](https://www.anaconda.com/products/individual)

Installation is preferably done in a virtual environment

## Installation

### Linux/MacOS
```
$ git clone https://github.com/mfreer/SPIFpy.git
$ cd /path/to/spifpy
$ pip install .
```
For those who are actively developing NRC-SPIFpy, you can change the last line to ```pip install --editable .```. 

### Windows

```
$ cd /path/to/spifpy
$ conda env create -f environment.yml
$ conda activate spifpy
$ pip install .
```

<a name="usage"></a>
## Example usage with 2DS imaging probe (SPEC Inc.)

1. Copy over required configuration files using `nrc-spifpy-cc`, and make any desired modifications to the config files. In this
case, the config files will include `2DS.ini` which defines config options for extracting and storing 2DS data, and
also `aux_config.ini`, which specifies configuration options for adding auxiliary data.

```
$ nrc-spifpy-cc 2DS
```

1. Process the file of interest using `nrc-spifpy-extract`

```
$ nrc-spifpy-extract example_file.2DS 2DS.ini 
```

3. Add auxiliary information to the **SPIF** file using `spifaddaux`(optional), but only for the
`2DS-V` dataset.

```
$ nrc-spifpy-addaux example_file_2DS.nc auxiliary_file.nc -i 2DS-V -c aux_config.ini 
```

<a name="supported-probes"></a>
### Supported probes

Currently the following Optical Array Probes (OAP) are supported:

- 2DC (Two Dimension Cloud particle imaging probe)
- 2DP (Two Dimension Precipitation particle imaging probe)
- 2DS (2D-Stereo, SPEC Inc.)
- CIP (Cloud Imaging Probe, DMT)
- PIP (Precipitation Imaging Probe, DMT)
- HVPS (High Volume Precipitation Spectrometer, SPEC Inc.)

<a name="citation"></a>
### Citations  
- <i>Bala, K., Freer, M., Bliankinshtein, N., Nichman, L., Shilin, S. and Wolde, M.: Standardized Imaging Probe Format and Algorithms: Implementation and Applications, 18th International Conference on Clouds and Precipitation (ICCP), Pune, India, 2-6 August, 2021.</i>
- <i>Single Particle Image Format (SPIF) conversion utility, https://doi.org/10.4224/40002712, 2021</i>

<a name="acknowledgment"></a>
### Acknowledgments
We acknowledge CloudSci LLC for the support in developing this tool
