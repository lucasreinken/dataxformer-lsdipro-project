# DataXFormer

This project focuses on implementing the DataXFormer architecture, a system designed to automatically discover and apply robust data transformations. It aims to support data cleaning, integration, and preprocessing workflows by identifying transformation patterns directly from example data.

## Description

This repository contains an implementation of the DataXFormer architecture as introduced in “DataXFormer: A Robust Transformation Discovery System” by Abedjan et al. The goal of the system is to learn how input data should be transformed based on example pairs and then generalize these transformations to unseen data. By discovering mapping rules, text patterns, normalization steps, and structural conversions, the architecture helps automate a large part of the data preparation pipeline.

The implementation was developed as part of the LSDI (Large Scale Data Integration) project at TU Berlin during the Winter Semester 2025/26, focusing on making transformation discovery reproducible, modular, and extensible for research and teaching.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Directory Structure

```text
project/
│
├── pyproject.toml          # dependencies & project metadata
├── uv.lock                 # reproducible environment
├── README.md
│
├── src/
│   └── dataxformer/
│
├── notebooks/
│
├── configs/
│   ├── ingestion.yaml
│   ├── discovery.yaml
│   └── evaluation.yaml
│
└── tests/

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
