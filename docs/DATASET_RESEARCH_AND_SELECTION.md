# Dataset research and selection

Research refresh: 2026-11-09. No external dataset was downloaded or added to
the repository during this phase. Raw data must remain outside Git and must be
acquired only under its publisher's terms.

| Candidate | What the primary source establishes | Sensor/text pairing | Decision |
|---|---|---|---|
| Microsoft Azure Predictive Maintenance | Five linked CSV tables: hourly telemetry, errors, maintenance, failures, and machine metadata. The Microsoft workshop notebook is the source reference. | Machine/time keys exist, but the records are structured event codes, not technician free text. The dataset is a benchmark/sample corpus, not evidence of MaintainIQ's real plant logs. | Useful for a separately namespaced structured-event adapter; not a genuine NLP fusion corpus. |
| IBM AssetOpsBench | IBM's repository and dataset card describe an Industry 4.0 agent benchmark with authored scenarios and operational records. The dataset card is Apache-2.0. | Scenarios and benchmark records are not a verified paired sensor stream plus technician-log corpus for this project. | Useful for assistant/agent research; not selected for fusion training. |
| MATWI | KU Leuven EAVISE describes MATWI as the Multimodal Automatic Tool Wear Inspection dataset. | Tool-wear inspection modality is not established as technician free text aligned to MaintainIQ equipment and prediction cutoffs. | Not selected without a schema/license/alignment review of the downloadable package. |
| Factory Sensors with Failure Labels | The publisher describes a CC0 multi-table, six-machine dataset with telemetry, failures, and work orders. | It is explicitly synthetic/generated and therefore cannot establish genuine industrial technician-log evidence. | Optional synthetic integration fixture only; not a genuine result source. |
| Industrial Maintenance Synthetic Dataset | The dataset card identifies roughly 2M generated sensor-tag/work-order rows, Apache-2.0, with intentional impurities. | Equipment IDs and dates are generated; descriptions are synthetic work-order text. | Suitable for parser and robustness tests only, never genuine performance claims. |
| Swedish paper-machine vibration annotations | LTU/SND describes 21,090 vibration/annotation pairs from two paper machines, with Swedish expert annotations and a ten-day historical association. | This is the strongest public sensor-language candidate found. The published representation has signal spectra, notes, IDs, and precomputed embeddings, but no clear per-sensor/machine-part distinction and the ten-day association must be treated as the publisher's label policy, not automatically leakage-safe for this project. | Candidate for a separately approved research intake, subject to license, download terms, Swedish NLP model availability, and an explicit cutoff audit. |

## Recommendation

No candidate was accepted as an authorized MaintainIQ production paired dataset
in this run. The Swedish dataset is the best research lead because the source
explicitly reports real paper-machine signals paired with expert technical
language, but it still requires a formal license/provenance and temporal-cutoff
review before ingestion or training. Azure remains useful for structured sensor
and event adapter tests, but its error/component codes must not be relabeled as
free-text technician logs.

The repository therefore keeps genuine multimodal training **blocked**. Existing
synthetic fixtures remain explicitly namespaced and are used only for contract
and smoke tests.

## Acquisition checklist

Before placing any raw data under `Data/external_datasets/<name>/`, record the
publisher URL, license, version, checksums, access approval, machine identity
semantics, timestamp semantics, text language, availability timestamp, and
prediction cutoff. Raw files are immutable; adapters write normalized outputs to
a separate versioned directory.

Primary references:

- Microsoft Azure workshop notebook: <https://github.com/microsoft/sqlworkshops/blob/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/notebooks/Predictive%20Maintenance%20in%20Python%20Notebook.ipynb>
- IBM AssetOpsBench: <https://github.com/IBM/AssetOpsBench>
- AssetOpsBench dataset card: <https://huggingface.co/datasets/ibm-research/AssetOpsBench>
- MATWI: <https://iiw.kuleuven.be/onderzoek/eavise/MATWI>
- Swedish vibration/technical-language dataset record: <https://ltu.diva-portal.org/smash/record.jsf?pid=diva2%3A1950819>
- Synthetic factory sensor dataset: <https://www.misata.studio/datasets/factory-sensors-failure-labels>
- Industrial Maintenance Synthetic dataset card: <https://huggingface.co/datasets/Jvachier/industrial-maintenance-synthetic>
