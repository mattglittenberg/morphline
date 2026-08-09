/*
 * Scanner harmonization and the scanner/time confound diagnostics (§2.3).
 *
 * Emits harmonization.json alongside the Parquet so the report stage can be a
 * separate process rather than needing the live result object.
 */
process HARMONIZE {
    tag 'harmonize'
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.{parquet,json}'

    input:
    path observations
    path config

    output:
    path 'harmonized_observations.parquet', emit: parquet
    path 'harmonization.json'             , emit: json
    path 'versions.yml'                   , emit: versions

    script:
    """
    morphline harmonize --config ${config} --observations ${observations} --outdir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
