/*
 * Parse one subject's stats files into canonical Parquet.
 *
 * This is the fan-out point: one task per subject. Each task emits the PATH of
 * the Parquet it wrote, never the table itself (§2.7).
 */
process PARSE_SUBJECT {
    tag "${subject_id}"
    publishDir "${params.outdir}/per_subject", mode: 'copy', pattern: '*.parquet'

    input:
    tuple val(subject_id), path(subject_dir)
    path config
    path dataset

    output:
    tuple val(subject_id), path("${subject_id}.observations.parquet")   , emit: parquet
    tuple val(subject_id), path("${subject_id}.parse_failures.parquet") , emit: failures
    tuple val(subject_id), path('versions.yml')                         , emit: versions

    script:
    """
    morphline ingest \\
        --config ${config} \\
        --indir ${dataset} \\
        --subject ${subject_id} \\
        --outdir .

    mv observations.parquet   ${subject_id}.observations.parquet
    mv parse_failures.parquet ${subject_id}.parse_failures.parquet

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
