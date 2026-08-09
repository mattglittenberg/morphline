/*
 * Longitudinal mixed-effects model with per-region convergence reporting (§2.5).
 */
process MODEL {
    tag 'model'
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.{parquet,json}'

    input:
    path observations
    path config

    output:
    path 'model_results.parquet', emit: parquet
    path 'model_results.json'   , emit: json
    path 'versions.yml'         , emit: versions

    script:
    """
    morphline model --config ${config} --observations ${observations} --outdir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
