# AneurysmTool

**Fluid–structure interaction for functional assessment of ascending aortic aneurysms:
a biomechanical approach toward clinical practice**

Fundação para a Ciência e a Tecnologia, PTDC/EMD-EMD/1230/2021
([10.54499/PTDC/EMD-EMD/1230/2021](https://doi.org/10.54499/PTDC/EMD-EMD/1230/2021)) ·
[project website](https://userweb.fct.unl.pt/~jmc.xavier/AneurysmTool/index.html)

AneurysmTool builds patient-specific computational models of the ascending thoracic aorta from
clinical imaging, so that wall mechanics and haemodynamics can be assessed without invasive
measurement. The project works along four pillars — patient-specific FSI modelling, ex-vivo tissue
characterisation, uncertainty quantification, and clinical decision support.

This repository holds the software released by the project. Each component sits in its own
directory with its own documentation and environment specification.

## Components

| Directory | What it does | Status |
|---|---|---|
| [`aorta-mesh-morphing/`](aorta-mesh-morphing/) | Boundary-condition generation and parameter calibration for mesh-morphing CFD of the thoracic aorta: cine-CTA wall-motion tracking, inlet velocity from 4D flow MRI, wall motion by RBF morphing, Windkessel outlet calibration, wall-stiffness calibration | Pre-release |

Components are added as they are released.

## Publications

| Year | Publication | DOI |
|---|---|---|
| 2026 | Valente R., Mourato A., Carvalho A., Xavier J., Brito M., Avril S., Tomás A., Fragata J. *Patient-specific in-vivo dynamic motion of ascending thoracic aortic aneurysms from cine CTA.* IEEE Transactions on Biomedical Engineering — its wall-motion tracking framework is released here, in `aorta-mesh-morphing/`, for the first time | [10.1109/TBME.2026.3681119](https://doi.org/10.1109/TBME.2026.3681119) |
| 2022 | Valente R., Mourato A., Brito M., Xavier J., Tomás A., Avril S. *Fluid–structure interaction modeling of ascending thoracic aortic aneurysms in SimVascular.* Biomechanics 2(2), 189–204 | [10.3390/biomechanics2020016](https://doi.org/10.3390/biomechanics2020016) |

A manuscript on mesh-morphing CFD driven by measured wall kinematics, comparing rigid-wall,
image-prescribed and fully coupled FSI models, is in preparation for *Computer Methods and Programs
in Biomedicine*. It will be listed here on publication.

## Data availability

Patient imaging and patient-derived inputs are not distributed by this project. They were acquired
under ethical approval that does not permit public redistribution. This includes derived geometry,
physiological boundary conditions, case transformations and study solver decks. Reusable code,
neutral solver templates and wholly synthetic worked examples are released openly. The public
archive therefore supports inspection and synthetic execution but not reproduction of the
patient-specific results.

## Project team and partners

José Xavier (PI, NOVA-SST) · José Fragata (Co-PI, NMS-FCM) · Stéphane Avril
(international partner, Mines Saint-Étienne)

NOVA.ID.FCT (lead) · NMS-FCM (clinical) · INEGI (experimental) · CEMAPRE-ISEG (statistics) ·
Mines Saint-Étienne (international)

Authorship of each released component is recorded in its own documentation and in
[`CITATION.cff`](CITATION.cff).

Contact: José Xavier, jmc.xavier@fct.unl.pt

## Releases and DOIs

Releases of this repository are archived by Zenodo, which assigns two identifiers: a *version DOI*
naming one immutable snapshot, and a *concept DOI* that always resolves to the most recent version.

**Cite the version DOI**, and cite the one for the release the results came from. A version DOI
identifies exactly the code that was archived at that tag: later components added to this
repository appear in later versions and cannot alter it. The concept DOI is not a substitute,
because what it resolves to changes as the project grows.

For a paper, give the version DOI together with the path to the component in the living
repository, so that a reader has both the fixed artefact and the maintained code.

## Licence and citation

Code is licensed under the Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Example data, once included, is CC-BY-4.0.

Citation metadata is in [`CITATION.cff`](CITATION.cff).

## Funding

- FCT, AneurysmTool, [PTDC/EMD-EMD/1230/2021](https://doi.org/10.54499/PTDC/EMD-EMD/1230/2021)
- UNIDEMI, [UID/00667/2025](https://doi.org/10.54499/UID/00667/2025)
- FCT PhD grant 2022.12223.BD (R. Valente)
- CUF Academic Center clinical research grant, 2024
