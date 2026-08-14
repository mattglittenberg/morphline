#!/usr/bin/env nextflow
/*
 * morphline — BIDS-aware longitudinal neuroimaging derivatives pipeline.
 *
 * BUILD_PLAN §2.7. DSL2, nf-core *conventions* without the full template:
 * --outdir, nextflow_schema.json, per-process versions.yml, and a
 * `-profile test` that runs on generated fixtures with no external data.
 *
 * CHANNELS CARRY FILE PATHS, NOT DATASETS.
 * Every process writes a Parquet file and emits its path; gathering steps
 * collect() paths and the consuming process reads them with pyarrow inside
 * its own container. No process emits a serialized dataframe, a large value
 * channel, or an in-memory table. This costs nothing at v1 scale and means
 * the architecture does not need rewriting if the dataset grows by two orders
 * of magnitude — which is a better answer in an interview than "it fit in
 * memory".
 */

nextflow.enable.dsl = 2

include { GENERATE_FIXTURES } from './modules/local/generate_fixtures'
include { PARSE_SUBJECT     } from './modules/local/parse_subject'
include { COLLECT_CANONICAL } from './modules/local/collect_canonical'
include { QC                } from './modules/local/qc'
include { HARMONIZE         } from './modules/local/harmonize'
include { MODEL             } from './modules/local/model'
include { ACCOUNTING        } from './modules/local/accounting'
include { REPORT            } from './modules/local/report'

def helpMessage() {
    log.info """
    morphline ${workflow.manifest.version}

    Usage:
      nextflow run . -profile test,docker --outdir results

    Options:
      --config      Path to the morphline YAML run configuration
      --input       Existing dataset root. Omitted under -profile test, which
                    generates synthetic fixtures instead
      --outdir      Output directory (default: results)
    """.stripIndent()
}

workflow {
    if (params.help) {
        helpMessage()
        return
    }

    ch_config = Channel.fromPath(params.config, checkIfExists: true).first()

    // Under the test profile there is deliberately no external data: fixtures
    // are generated as the first process, so a cold clone runs end to end.
    if (params.input) {
        ch_dataset = Channel.fromPath(params.input, type: 'dir', checkIfExists: true).first()
    }
    else {
        GENERATE_FIXTURES(ch_config)
        ch_dataset = GENERATE_FIXTURES.out.fixtures
    }

    // Fan out over subjects. Each subject directory is parsed independently,
    // which is where real parallelism lives once a dataset is large.
    ch_subjects = ch_dataset
        .flatMap { root -> file("${root}/derivatives/freesurfer").listFiles().findAll { it.isDirectory() } }
        .map { dir -> tuple(dir.name, dir) }

    PARSE_SUBJECT(ch_subjects, ch_config, ch_dataset)

    // Gather PATHS, not tables. The consuming process does the reading. The
    // ingest sidecars are gathered alongside the observations: they fan out
    // per subject and the accounting and provenance stages need them merged.
    COLLECT_CANONICAL(
        PARSE_SUBJECT.out.parquet.map { _id, pq -> pq }.collect(),
        PARSE_SUBJECT.out.counters.map { _id, c -> c }.collect(),
        PARSE_SUBJECT.out.observed_versions.map { _id, v -> v }.collect(),
        ch_config,
    )

    QC(COLLECT_CANONICAL.out.parquet, ch_config)
    HARMONIZE(QC.out.parquet, ch_config)
    // Both arms reach MODEL: the harmonized values it fits, and the
    // pre-harmonization values the §2.3.1 sensitivity comparison needs.
    MODEL(HARMONIZE.out.parquet, QC.out.parquet, ch_config)

    ACCOUNTING(
        COLLECT_CANONICAL.out.parquet,
        COLLECT_CANONICAL.out.counters,
        PARSE_SUBJECT.out.failures.map { _id, f -> f }.collect(),
        QC.out.parquet,
        MODEL.out.parquet,
        ch_config,
        ch_dataset,
    )

    REPORT(
        ACCOUNTING.out.funnel,
        ACCOUNTING.out.json,
        QC.out.parquet,
        COLLECT_CANONICAL.out.observed_versions,
        HARMONIZE.out.parquet,
        HARMONIZE.out.json,
        MODEL.out.parquet,
        MODEL.out.json,
        ch_config,
        ch_dataset,
    )

    // nf-core convention: one versions.yml per process, merged at the end.
    ch_versions = PARSE_SUBJECT.out.versions.map { _id, v -> v }.first()
        .mix(COLLECT_CANONICAL.out.versions)
        .mix(QC.out.versions)
        .mix(HARMONIZE.out.versions)
        .mix(MODEL.out.versions)
        .mix(ACCOUNTING.out.versions)
        .mix(REPORT.out.versions)

    ch_versions.collectFile(name: 'versions.yml', storeDir: "${params.outdir}/pipeline_info")
}
