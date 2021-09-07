# SPIFpy

Single Particle Image Format (SPIF) conversion utility.

## About

**SPIFpy** is a set of Command Line Interface(CLI) tools which allow for the conversion of files stored in a 
variety of raw imaging probe formats to the **SPIF** format. The package is written in **Python**, 
and includes the following utilities:

- `spifpy`: Convert a file in a raw imaging probe format to the **SPIF** format.
- `spifaddaux`: Add auxiliary data to a file in the **SPIF** format.
- `spifcc`: Copy the configuration files required for processing with `spifpy` and `spifaddaux`.

## Installation Requirements

- Python 3.6+

## Installation

```
$ git clone https://github.com/mfreer/SPIFpy.git
$ cd /path/to/spifpy
$ pip install .
```

Please note that for non Linux/MacOS distributions, the host computer will need to be installed using the Anaconda Python distribution using the same command.

<a name="usage"></a>
## Usage

1. Copy over required configuration files using `spifcc`, and make any desired modifications to the config files. In this
case, the config files will include `2DS.ini` which defines config options for extracting and storing 2DS data, and
also `aux_config.ini`, which specifies configuration options for adding auxiliary data.

```
$ spifcc 2DS
```

1. Process the file of interest using `spifpy`

```
$ spifpy example_file.2DS 2DS.ini 
```

3. Add auxiliary information to the **SPIF** file using `spifaddaux`(optional), but only for the
`2DS-V` dataset.

```
$ spifaddaux example_file_2DS.nc auxiliary_file.nc -i 2DS-V -c aux_config.ini 
```

<a name="supported-probes"></a>
### Supported probes

Currently the following probe types are supported:

- 2DS
- 2DC
- HVPS
- PIP
- CIP
