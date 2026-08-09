/*
 * Self-contained HTML report with the full provenance block (§2.8).
 */
process REPORT {
    tag 'report'
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.{html,json}'

    input:
    path accounting_funnel
    path qc_observations
    path harmonized
    path model_results
    path config
    path dataset

    output:
    path 'report.html'     , emit: html
    path 'provenance.json' , emit: provenance
    path 'versions.yml'    , emit: versions

    script:
    """
    morphline provenance --config ${config} --observations ${qc_observations} --outdir .
    morphline report --config ${config} --indir . --outdir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
