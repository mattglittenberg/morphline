/*
 * Gather per-subject Parquet files into one canonical table.
 *
 * The channel delivered PATHS; the reading happens here, inside the consuming
 * process, with pyarrow (§2.7). No dataframe ever travelled through Nextflow.
 */
process COLLECT_CANONICAL {
    tag 'collect'
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.parquet'

    input:
    path per_subject_parquet
    path config

    output:
    path 'observations.parquet', emit: parquet
    path 'versions.yml'        , emit: versions

    script:
    """
    morphline collect ${per_subject_parquet} --outdir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
