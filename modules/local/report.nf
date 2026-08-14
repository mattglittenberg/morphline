/*
 * Self-contained HTML report with the full provenance block (§2.8).
 */
process REPORT {
    tag 'report'
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.{html,json}'

    // `morphline report --indir .` reads four JSON sidecars from the staged
    // directory. Nextflow gives each task its own work dir, so every sidecar
    // must be a declared input — the in-process run finds them because every
    // stage writes to one shared outdir, which is exactly the difference this
    // process has to bridge. provenance.json is the only one not staged: the
    // preceding command writes it here.
    input:
    path accounting_funnel
    path accounting_json
    path qc_observations
    path harmonized
    path harmonization_json
    path model_results
    path model_json
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
