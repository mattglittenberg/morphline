/*
 * Gather per-subject Parquet files into one canonical table.
 *
 * The channel delivered PATHS; the reading happens here, inside the consuming
 * process, with pyarrow (§2.7). No dataframe ever travelled through Nextflow.
 */
process COLLECT_CANONICAL {
    tag 'collect'
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.{parquet,json}'

    input:
    path per_subject_parquet
    path per_subject_counters
    path per_subject_versions
    path config

    output:
    path 'observations.parquet' , emit: parquet
    path 'ingest_counters.json' , emit: counters
    path 'ingest_versions.json' , emit: observed_versions
    path 'versions.yml'         , emit: versions

    // Sidecar paths are passed explicitly. Deriving them from the inputs'
    // parents works only when each input has its own directory, which is true
    // of the staged-CLI parity harness and false here: Nextflow stages all of
    // them into this one work dir.
    script:
    def counter_args = per_subject_counters.collect { "--counters ${it}" }.join(' ')
    def version_args = per_subject_versions.collect { "--versions ${it}" }.join(' ')
    """
    morphline collect ${per_subject_parquet} ${counter_args} ${version_args} --outdir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
