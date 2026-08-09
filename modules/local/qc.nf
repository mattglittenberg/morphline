/*
 * Assign three-level QC status and the analysis inclusion decision (§2.4.1).
 */
process QC {
    tag 'qc'
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.parquet'

    input:
    path observations
    path config

    output:
    path 'qc_observations.parquet', emit: parquet
    path 'versions.yml'           , emit: versions

    script:
    """
    morphline qc --config ${config} --observations ${observations} --outdir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
